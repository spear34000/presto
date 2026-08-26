#include "presto/runtime/gguf.hpp"

#include <limits>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace presto::runtime {

struct MappedFile::Impl {
  const std::byte* data = nullptr;
  std::uint64_t size = 0;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
#else
  int file = -1;
#endif

  ~Impl() {
#if defined(_WIN32)
    if (data) UnmapViewOfFile(data);
    if (mapping) CloseHandle(mapping);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (data) munmap(const_cast<std::byte*>(data), static_cast<std::size_t>(size));
    if (file >= 0) close(file);
#endif
  }
};

MappedFile::MappedFile() = default;
MappedFile::~MappedFile() = default;
MappedFile::MappedFile(MappedFile&&) noexcept = default;
MappedFile& MappedFile::operator=(MappedFile&&) noexcept = default;

Result<MappedFile> MappedFile::open_readonly(const std::string& path) {
  MappedFile result;
  result.impl_ = std::make_unique<Impl>();
#if defined(_WIN32)
  const int wide_count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(),
                                              static_cast<int>(path.size()), nullptr, 0);
  if (wide_count <= 0) {
    return Result<MappedFile>::failure(Status::error(
        StatusCode::InvalidArgument, "map_file", "path is not valid UTF-8"));
  }
  std::wstring wide(static_cast<std::size_t>(wide_count), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), static_cast<int>(path.size()),
                      wide.data(), wide_count);
  result.impl_->file = CreateFileW(wide.c_str(), GENERIC_READ,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL, nullptr);
  if (result.impl_->file == INVALID_HANDLE_VALUE) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::NotFound, "map_file", "cannot open file"));
  }
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(result.impl_->file, &length) || length.QuadPart <= 0) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::InvalidArgument, "map_file", "file is empty or unreadable"));
  }
  result.impl_->size = static_cast<std::uint64_t>(length.QuadPart);
  result.impl_->mapping = CreateFileMappingW(result.impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!result.impl_->mapping) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::AllocationFailed, "map_file", "CreateFileMappingW failed"));
  }
  result.impl_->data = static_cast<const std::byte*>(
      MapViewOfFile(result.impl_->mapping, FILE_MAP_READ, 0, 0, 0));
#else
  result.impl_->file = ::open(path.c_str(), O_RDONLY);
  if (result.impl_->file < 0) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::NotFound, "map_file", "cannot open file"));
  }
  struct stat stat_buffer {};
  if (fstat(result.impl_->file, &stat_buffer) != 0 || stat_buffer.st_size <= 0) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::InvalidArgument, "map_file", "file is empty or unreadable"));
  }
  result.impl_->size = static_cast<std::uint64_t>(stat_buffer.st_size);
  if (result.impl_->size > std::numeric_limits<std::size_t>::max()) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::Unsupported, "map_file", "file is too large for address space"));
  }
  const void* mapped = mmap(nullptr, static_cast<std::size_t>(result.impl_->size), PROT_READ,
                            MAP_PRIVATE, result.impl_->file, 0);
  result.impl_->data = mapped == MAP_FAILED ? nullptr : static_cast<const std::byte*>(mapped);
#endif
  if (!result.impl_->data) {
    return Result<MappedFile>::failure(
        Status::error(StatusCode::AllocationFailed, "map_file", "read-only mapping failed"));
  }
  return Result<MappedFile>::success(std::move(result));
}

std::uint64_t MappedFile::size() const { return impl_ ? impl_->size : 0; }

Result<std::span<const std::byte>> MappedFile::bytes(const std::uint64_t offset,
                                                     const std::uint64_t length) const {
  if (!impl_ || offset > impl_->size || length > impl_->size - offset ||
      length > std::numeric_limits<std::size_t>::max()) {
    return Result<std::span<const std::byte>>::failure(Status::error(
        StatusCode::InvalidArgument, "mapped_bytes", "requested byte span is out of bounds"));
  }
  return Result<std::span<const std::byte>>::success(
      {impl_->data + static_cast<std::size_t>(offset), static_cast<std::size_t>(length)});
}

Result<GgufWeights> GgufWeights::open(const std::string& path) {
  auto index = read_gguf_index(path);
  if (!index.ok()) return Result<GgufWeights>::failure(index.status);
  auto mapping = MappedFile::open_readonly(path);
  if (!mapping.ok()) return Result<GgufWeights>::failure(mapping.status);
  GgufWeights weights;
  weights.index_ = std::move(index.value);
  weights.mapping_ = std::move(mapping.value);
  return Result<GgufWeights>::success(std::move(weights));
}

Result<std::span<const std::byte>> GgufWeights::tensor_bytes(const std::string_view name) const {
  const auto* tensor = index_.find_tensor(name);
  if (!tensor) {
    return Result<std::span<const std::byte>>::failure(
        Status::error(StatusCode::NotFound, "tensor_bytes", "GGUF tensor was not found"));
  }
  return mapping_.bytes(tensor->absolute_offset, tensor->byte_size);
}

} // namespace presto::runtime
