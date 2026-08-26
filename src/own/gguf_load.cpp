// presto own-engine: standalone GGUF v2/v3 loader with f32 dequantization.
// Pure C++20, zero ggml/llama.cpp dependency. Spec: ggml docs/gguf.md.
//
// Supported tensor encodings: F32, F16, BF16, Q4_0, Q8_0, Q4_K (the Q4_K_M
// container uses this type id), Q6_K. Any other type fails load() with an
// error naming the type id and its ggml name.
//
// Version handling: v1 files use u32 string lengths / counts / offsets,
// v2 and v3 use u64 widths (v2 and v3 are structurally identical). All
// integers are little-endian; little-endian host assumed.
//
// Define OWN_GGUF_SELFTEST to additionally compile a tiny CLI main that
// loads a model path from argv[1], prints the tensor count plus the first 3
// tensor names, and smoke-tests encode/decode. Never compiled in normal
// builds.
#include "own.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace presto::own {
namespace {

#ifdef OWN_GGUF_SELFTEST
// Self-test hook: tensor names of the last successful load (private members
// are not otherwise reachable from the selftest main below).
std::vector<std::string> g_selftest_names;
#endif

// ---------------------------------------------------------------------------
// ggml type ids (subset we dequantize)
// ---------------------------------------------------------------------------
constexpr uint32_t kT_F32 = 0;
constexpr uint32_t kT_F16 = 1;
constexpr uint32_t kT_Q4_0 = 2;
constexpr uint32_t kT_Q8_0 = 8;
constexpr uint32_t kT_Q4_K = 12;  // Q4_K_M tensors carry this id
constexpr uint32_t kT_Q6_K = 14;
constexpr uint32_t kT_BF16 = 30;

struct TypeShape {
  uint32_t vals;   // values per block
  uint32_t bytes;  // bytes per block on disk
};

bool type_shape(uint32_t t, TypeShape& s) {
  switch (t) {
    case kT_F32:  s = {1, 4};     return true;
    case kT_F16:  s = {1, 2};     return true;
    case kT_BF16: s = {1, 2};     return true;
    case kT_Q4_0: s = {32, 18};   return true;
    case kT_Q8_0: s = {32, 34};   return true;
    case kT_Q4_K: s = {256, 144}; return true;
    case kT_Q6_K: s = {256, 210}; return true;
    default: return false;
  }
}

const char* type_name(uint32_t t) {
  switch (t) {
    case 0: return "F32";     case 1: return "F16";     case 2: return "Q4_0";
    case 3: return "Q4_1";    case 6: return "Q5_0";    case 7: return "Q5_1";
    case 8: return "Q8_0";    case 9: return "Q8_1";    case 10: return "Q2_K";
    case 11: return "Q3_K";   case 12: return "Q4_K";   case 13: return "Q5_K";
    case 14: return "Q6_K";   case 15: return "Q8_K";   case 16: return "IQ2_XXS";
    case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 19: return "IQ1_S";
    case 20: return "IQ4_NL"; case 21: return "IQ3_S";  case 22: return "IQ2_S";
    case 23: return "IQ4_XS"; case 24: return "I8";     case 25: return "I16";
    case 26: return "I32";    case 27: return "I64";    case 28: return "F64";
    case 29: return "IQ1_M";  case 30: return "BF16";
    default: return "UNKNOWN";
  }
}

// ---------------------------------------------------------------------------
// little-endian cursor over the file image
// ---------------------------------------------------------------------------
struct Rd {
  const unsigned char* p = nullptr;
  const unsigned char* end = nullptr;
  bool bad = false;

  std::size_t left() const { return bad ? 0 : static_cast<std::size_t>(end - p); }
  bool need(std::size_t n) {
    if (!bad && left() >= n) return true;
    bad = true;
    return false;
  }
  uint8_t u8() {
    if (!need(1)) return 0;
    return *p++;
  }
  uint16_t u16() {
    if (!need(2)) return 0;
    const uint16_t v = static_cast<uint16_t>(p[0] | (static_cast<unsigned>(p[1]) << 8));
    p += 2;
    return v;
  }
  uint32_t u32() {
    if (!need(4)) return 0;
    const uint32_t v = static_cast<uint32_t>(p[0]) |
                       (static_cast<uint32_t>(p[1]) << 8) |
                       (static_cast<uint32_t>(p[2]) << 16) |
                       (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return v;
  }
  uint64_t u64() {
    if (!need(8)) return 0;
    const uint64_t lo = u32();
    const uint64_t hi = u32();
    return lo | (hi << 32);
  }
  float f32() {
    const uint32_t b = u32();
    float f = 0.f;
    std::memcpy(&f, &b, sizeof f);
    return f;
  }
  double f64() {
    const uint64_t b = u64();
    double d = 0.0;
    std::memcpy(&d, &b, sizeof d);
    return d;
  }
};

uint16_t le16(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (static_cast<unsigned>(p[1]) << 8));
}

// GGUF string: u64 length (u32 in v1) + raw bytes.
bool rd_string(Rd& r, bool narrow_counts, std::string& out) {
  const uint64_t len = narrow_counts ? r.u32() : r.u64();
  if (r.bad) return false;
  if (r.left() < len) {
    r.bad = true;
    return false;
  }
  out.assign(reinterpret_cast<const char*>(r.p), static_cast<std::size_t>(len));
  r.p += len;
  return true;
}

// ---------------------------------------------------------------------------
// metadata value types
// ---------------------------------------------------------------------------
enum : uint32_t {
  kV_U8 = 0, kV_I8 = 1, kV_U16 = 2, kV_I16 = 3, kV_U32 = 4, kV_I32 = 5,
  kV_F32 = 6, kV_BOOL = 7, kV_STR = 8, kV_ARR = 9, kV_U64 = 10, kV_I64 = 11,
  kV_F64 = 12,
};

uint32_t scalar_width(uint32_t t) {
  switch (t) {
    case kV_U8: case kV_I8: case kV_BOOL: return 1;
    case kV_U16: case kV_I16: return 2;
    case kV_U32: case kV_I32: case kV_F32: return 4;
    case kV_U64: case kV_I64: case kV_F64: return 8;
    default: return 0;  // variable length or invalid
  }
}

// Skip one value of the given type. Arrays of scalars skip by stride, arrays
// of strings walk element by element.
bool skip_value(Rd& r, bool narrow_counts, uint32_t t,
                uint64_t* arr_count_out = nullptr) {
  if (t != kV_ARR) {
    const uint32_t w = scalar_width(t);
    if (w == 0) {
      r.bad = true;
      return false;
    }
    return r.need(w) && (r.p += w, true);
  }
  const uint32_t et = r.u32();
  const uint64_t n = narrow_counts ? r.u32() : r.u64();
  if (r.bad) return false;
  if (arr_count_out) *arr_count_out = n;
  if (et == kV_ARR) {  // spec forbids nested arrays
    r.bad = true;
    return false;
  }
  const uint32_t w = scalar_width(et);
  if (w != 0) {
    if (n > r.left() / w) {  // overflow-safe stride check
      r.bad = true;
      return false;
    }
    r.p += static_cast<std::size_t>(n) * w;
    return true;
  }
  if (et != kV_STR) {
    r.bad = true;
    return false;
  }
  std::string tmp;
  for (uint64_t i = 0; i < n; ++i)
    if (!rd_string(r, narrow_counts, tmp)) return false;
  return true;
}

// Read one scalar value and render it as a canonical string for kv_ storage.
bool scalar_to_str(Rd& r, uint32_t t, std::string& out) {
  char tmp[48];
  tmp[0] = '\0';
  switch (t) {
    case kV_U8: {
      const uint32_t v = r.u8();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%u", v);
      break;
    }
    case kV_I8: {
      const int v = static_cast<int8_t>(r.u8());
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%d", v);
      break;
    }
    case kV_BOOL: {
      const uint8_t v = r.u8();
      if (r.bad) return false;
      out = v ? "true" : "false";
      return true;
    }
    case kV_U16: {
      const uint32_t v = r.u16();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%u", v);
      break;
    }
    case kV_I16: {
      const int v = static_cast<int16_t>(r.u16());
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%d", v);
      break;
    }
    case kV_U32: {
      const uint32_t v = r.u32();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%u", v);
      break;
    }
    case kV_I32: {
      const int v = static_cast<int32_t>(r.u32());
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%d", v);
      break;
    }
    case kV_U64: {
      const uint64_t v = r.u64();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%llu", static_cast<unsigned long long>(v));
      break;
    }
    case kV_I64: {
      const int64_t v = static_cast<int64_t>(r.u64());
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%lld", static_cast<long long>(v));
      break;
    }
    case kV_F32: {
      const float v = r.f32();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%.9g", static_cast<double>(v));
      break;
    }
    case kV_F64: {
      const double v = r.f64();
      if (r.bad) return false;
      std::snprintf(tmp, sizeof tmp, "%.17g", v);
      break;
    }
    default:
      return false;
  }
  out = tmp;
  return true;
}

// ---------------------------------------------------------------------------
// float conversions
// ---------------------------------------------------------------------------
float fp16_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x03FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;  // signed zero
    } else {
      exp = 113;  // 127 - 15 + 1
      while ((man & 0x0400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x03FFu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float f = 0.f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

float bf16_to_f32(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

// ---------------------------------------------------------------------------
// dequantizers: src -> nvals floats (row-major, ne[0] fastest)
// ---------------------------------------------------------------------------
void dequant_f16(const unsigned char* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = fp16_to_f32(le16(src + 2 * i));
}

void dequant_bf16(const unsigned char* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(le16(src + 2 * i));
}

void dequant_q4_0(const unsigned char* src, float* dst, std::size_t n) {
  const std::size_t nb = n / 32;
  for (std::size_t b = 0; b < nb; ++b, src += 18) {
    const float d = fp16_to_f32(le16(src));
    const unsigned char* q = src + 2;
    float* y = dst + b * 32;
    for (int i = 0; i < 16; ++i) {
      y[i] = d * static_cast<float>(static_cast<int>(q[i] & 0xF) - 8);
      y[i + 16] = d * static_cast<float>(static_cast<int>(q[i] >> 4) - 8);
    }
  }
}

void dequant_q8_0(const unsigned char* src, float* dst, std::size_t n) {
  const std::size_t nb = n / 32;
  for (std::size_t b = 0; b < nb; ++b, src += 34) {
    const float d = fp16_to_f32(le16(src));
    const int8_t* q = reinterpret_cast<const int8_t*>(src + 2);
    float* y = dst + b * 32;
    for (int i = 0; i < 32; ++i) y[i] = d * static_cast<float>(q[i]);
  }
}

// 6-bit scale/min unpacking for K-quants (matches ggml get_scale_min_k4).
void get_scale_min_k4(int j, const unsigned char* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = static_cast<uint8_t>(q[j] & 63);
    m = static_cast<uint8_t>(q[j + 4] & 63);
  } else {
    d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// Q4_K superblock: 256 values in 144 bytes.
// layout: f16 d | f16 dmin | 12B packed 6-bit scales/mins | 128B nibbles.
// 8 sub-blocks of 32 values; nibble order low-then-high within each byte.
void dequant_q4_k(const unsigned char* src, float* dst, std::size_t n) {
  const std::size_t nb = n / 256;
  for (std::size_t b = 0; b < nb; ++b, src += 144) {
    const float d = fp16_to_f32(le16(src));
    const float dmin = fp16_to_f32(le16(src + 2));
    const unsigned char* scales = src + 4;
    const unsigned char* q = src + 16;
    float* y = dst + b * 256;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc = 0, m = 0;
      get_scale_min_k4(is + 0, scales, sc, m);
      const float d1 = d * sc;
      const float m1 = dmin * m;
      get_scale_min_k4(is + 1, scales, sc, m);
      const float d2 = d * sc;
      const float m2 = dmin * m;
      for (int l = 0; l < 32; ++l) y[j + l] = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) y[j + l + 32] = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// Q6_K superblock: 256 values in 210 bytes.
// layout: 128B low-qubits | 64B high-qubits | 16 x i8 scales | f16 d.
// 6-bit index: ql low/high nibbles combined with 2 qh bits, minus 32.
void dequant_q6_k(const unsigned char* src, float* dst, std::size_t n) {
  const std::size_t nb = n / 256;
  for (std::size_t b = 0; b < nb; ++b, src += 210) {
    const float d = fp16_to_f32(le16(src + 208));
    const unsigned char* ql = src;
    const unsigned char* qh = src + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(src + 192);
    float* y = dst + b * 256;
    for (int j = 0; j < 256; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 =
            static_cast<int8_t>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 =
            static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 =
            static_cast<int8_t>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 =
            static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[j + l] = d * static_cast<float>(sc[is]) * static_cast<float>(q1);
        y[j + l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[j + l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[j + l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// GGufModel::load
// ---------------------------------------------------------------------------
bool GGufModel::load(const std::string& path, GGufModel& out, std::string& err) {
  out = GGufModel{};

  // ---- read whole file into buf_ (single file image) ----
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    err = "cannot open '" + path + "'";
    return false;
  }
  const std::streamoff fsize = f.tellg();
  constexpr std::streamoff kMinHeader = 24;  // magic4 + version4 + counts 2x8
  if (fsize < kMinHeader) {
    err = "file too small to be a GGUF file: '" + path + "'";
    return false;
  }
  f.seekg(0);
  out.buf_.resize(static_cast<std::size_t>(fsize));
  f.read(out.buf_.data(), fsize);
  if (!f || f.gcount() != fsize) {
    err = "short read on '" + path + "'";
    return false;
  }

  const unsigned char* base = reinterpret_cast<const unsigned char*>(out.buf_.data());
  Rd r{base, base + out.buf_.size(), false};

  // ---- header ----
  if (!r.need(4) || std::memcmp(r.p, "GGUF", 4) != 0) {
    err = "bad magic: not a GGUF file";
    return false;
  }
  r.p += 4;
  const uint32_t version = r.u32();
  if (r.bad || version < 1 || version > 3) {
    err = "unsupported GGUF version " + std::to_string(version);
    return false;
  }
  const bool narrow_counts = (version == 1);  // v1: u32 lengths/counts/offsets
  const uint64_t n_tensors = narrow_counts ? r.u32() : r.u64();
  const uint64_t n_kv = narrow_counts ? r.u32() : r.u64();
  if (r.bad) {
    err = "truncated GGUF header";
    return false;
  }
  constexpr uint64_t kMaxCount = 1ull << 20;  // sanity caps against garbage headers
  if (n_tensors > kMaxCount || n_kv > kMaxCount) {
    err = "implausible header counts (tensors=" + std::to_string(n_tensors) +
          ", kv=" + std::to_string(n_kv) + ")";
    return false;
  }
  uint64_t tokens_array_count = 0;

  // ---- kv metadata pairs (all scalar types stored as strings; arrays skipped) ----
  out.kv_.reserve(static_cast<std::size_t>(n_kv));
  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key;
    if (!rd_string(r, narrow_counts, key)) {
      err = "truncated metadata key #" + std::to_string(i);
      return false;
    }
    const uint32_t vt = r.u32();
    if (r.bad) {
      err = "truncated metadata value type for '" + key + "'";
      return false;
    }
    if (vt == kV_STR) {
      std::string val;
      if (!rd_string(r, narrow_counts, val)) {
        err = "truncated metadata string value for '" + key + "'";
        return false;
      }
      out.kv_.emplace_back(std::move(key), std::move(val));
    } else if (vt == kV_ARR) {
      uint64_t arr_count = 0;
      if (!skip_value(r, narrow_counts, vt, &arr_count)) {
        err = "malformed or truncated metadata array '" + key + "'";
        return false;
      }
      if (key == "tokenizer.ggml.tokens") tokens_array_count = arr_count;
      // arrays not mirrored into kv_ (tokenizer reads them from buf_)
    } else {
      std::string val;
      if (!scalar_to_str(r, vt, val)) {
        err = "malformed metadata scalar '" + key + "' (type " + std::to_string(vt) + ")";
        return false;
      }
      out.kv_.emplace_back(std::move(key), std::move(val));
    }
  }

  // ---- tensor info table ----
  struct TInfo {
    std::string name;
    uint32_t type = 0;
    uint64_t offset = 0;
    std::vector<uint64_t> ne;
  };
  std::vector<TInfo> infos;
  infos.reserve(static_cast<std::size_t>(n_tensors));
  for (uint64_t i = 0; i < n_tensors; ++i) {
    TInfo ti;
    if (!rd_string(r, narrow_counts, ti.name)) {
      err = "truncated tensor name #" + std::to_string(i);
      return false;
    }
    const uint32_t n_dims = r.u32();
    if (r.bad) {
      err = "truncated tensor info for '" + ti.name + "'";
      return false;
    }
    if (n_dims == 0 || n_dims > 4) {
      err = "tensor '" + ti.name + "': unsupported dim count " + std::to_string(n_dims);
      return false;
    }
    ti.ne.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      ti.ne[d] = narrow_counts ? r.u32() : r.u64();
      if (r.bad) {
        err = "truncated dims for '" + ti.name + "'";
        return false;
      }
      if (ti.ne[d] == 0 || ti.ne[d] > 0xFFFFFFFFull) {
        err = "tensor '" + ti.name + "': bad dim " + std::to_string(ti.ne[d]);
        return false;
      }
    }
    ti.type = r.u32();
    ti.offset = narrow_counts ? r.u32() : r.u64();
    if (r.bad) {
      err = "truncated tensor info for '" + ti.name + "'";
      return false;
    }
    infos.push_back(std::move(ti));
  }

  // ---- alignment + data section start ----
  int64_t align_i = out.meta_int("general.alignment", 32);
  if (align_i < 1 || align_i > (1 << 20) || (align_i & (align_i - 1)) != 0) {
    err = "invalid general.alignment " + std::to_string(align_i);
    return false;
  }
  const uint64_t align = static_cast<uint64_t>(align_i);
  const uint64_t pos_after_infos = static_cast<uint64_t>(r.p - base);
  const uint64_t data_start =
      (pos_after_infos + align - 1) & ~(align - 1);  // align up
  if (data_start > out.buf_.size()) {
    err = "tensor data section starts past end of file";
    return false;
  }

  // ---- materialize every tensor dequantized to f32 ----
  out.tensors_.reserve(infos.size());
  for (const TInfo& ti : infos) {
    TypeShape ts{};
    if (!type_shape(ti.type, ts)) {
      err = "tensor '" + ti.name + "': unsupported ggml type " +
            std::to_string(ti.type) + " (" + type_name(ti.type) + ")";
      return false;
    }
    uint64_t nvals = 1;
    Tensor t;
    t.name = ti.name;
    t.ne.reserve(ti.ne.size());
    for (uint64_t d : ti.ne) {
      nvals *= d;
      if (nvals > (1ull << 33)) {  // > 16 GB of f32: reject early
        err = "tensor '" + ti.name + "': too large (" + std::to_string(nvals) +
              " elements)";
        return false;
      }
      t.ne.push_back(static_cast<uint32_t>(d));
    }
    if (nvals % ts.vals != 0) {
      err = "tensor '" + ti.name + "': element count " + std::to_string(nvals) +
            " not a multiple of block size " + std::to_string(ts.vals);
      return false;
    }
    const uint64_t nbytes = (nvals / ts.vals) * ts.bytes;
    const uint64_t avail = static_cast<uint64_t>(out.buf_.size()) - data_start;
    if (ti.offset > avail || nbytes > avail - ti.offset) {
      err = "tensor '" + ti.name + "': data range [" +
            std::to_string(ti.offset) + ", +" + std::to_string(nbytes) +
            ") outside file";
      return false;
    }

    t.data.resize(static_cast<std::size_t>(nvals));
    const unsigned char* src = base + data_start + ti.offset;
    float* dst = t.data.data();
    switch (ti.type) {
      case kT_F32:
        std::memcpy(dst, src, static_cast<std::size_t>(nvals) * 4);
        break;
      case kT_F16:
        dequant_f16(src, dst, static_cast<std::size_t>(nvals));
        break;
      case kT_BF16:
        dequant_bf16(src, dst, static_cast<std::size_t>(nvals));
        break;
      case kT_Q4_0:
        dequant_q4_0(src, dst, static_cast<std::size_t>(nvals));
        break;
      case kT_Q8_0:
        dequant_q8_0(src, dst, static_cast<std::size_t>(nvals));
        break;
      case kT_Q4_K:
        dequant_q4_k(src, dst, static_cast<std::size_t>(nvals));
        break;
      case kT_Q6_K:
        dequant_q6_k(src, dst, static_cast<std::size_t>(nvals));
        break;
      default:
        break;  // unreachable: type_shape gated above
    }
    out.tensors_.push_back(std::move(t));
  }

  // ---- llama-arch config from metadata ----
  out.arch = out.meta_str("general.architecture");
  const std::string pre = out.arch.empty() ? std::string("llama") : out.arch;
  out.n_vocab = static_cast<int>(out.meta_int(pre + ".vocab_size", 0));
  if (out.n_vocab <= 0) {
    // older conversions omit vocab_size: derive from the tokens array count
    out.n_vocab = static_cast<int>(tokens_array_count);
  }
  out.n_ctx_train = static_cast<int>(out.meta_int(pre + ".context_length", 0));
  out.n_embd = static_cast<int>(out.meta_int(pre + ".embedding_length", 0));
  out.n_layer = static_cast<int>(out.meta_int(pre + ".block_count", 0));
  out.n_head = static_cast<int>(out.meta_int(pre + ".attention.head_count", 0));
  out.n_head_kv =
      static_cast<int>(out.meta_int(pre + ".attention.head_count_kv", out.n_head));
  out.n_rot = static_cast<int>(out.meta_int(pre + ".rope.dimension_count", 0));
  out.ffn_dim = static_cast<int>(out.meta_int(pre + ".feed_forward_length", 0));
  out.rope_theta = out.meta_float(pre + ".rope.freq_base", 10000.f);
  out.norm_eps =
      out.meta_float(pre + ".attention.layer_norm_rms_epsilon", 1e-5f);
  // add_bos: honor tokenizer.ggml.add_bos_token when present (default true).
  for (const auto& kv : out.kv_) {
    if (kv.first == "tokenizer.ggml.add_bos_token") {
      out.add_bos = (kv.second == "true" || kv.second == "1");
      break;
    }
  }

#ifdef OWN_GGUF_SELFTEST
  g_selftest_names.clear();
  g_selftest_names.reserve(out.tensors_.size());
  for (const Tensor& t : out.tensors_) g_selftest_names.push_back(t.name);
#endif
  return true;
}

// ---------------------------------------------------------------------------
// accessors
// ---------------------------------------------------------------------------
const Tensor* GGufModel::tensor(const std::string& name) const {
  for (const Tensor& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

std::string GGufModel::meta_str(const std::string& key,
                                const std::string& dflt) const {
  for (const auto& kv : kv_)
    if (kv.first == key) return kv.second;
  return dflt;
}

int64_t GGufModel::meta_int(const std::string& key, int64_t dflt) const {
  for (const auto& kv : kv_) {
    if (kv.first != key) continue;
    if (kv.second == "true") return 1;
    if (kv.second == "false") return 0;
    char* endp = nullptr;
    const long long v = std::strtoll(kv.second.c_str(), &endp, 10);
    if (endp != kv.second.c_str()) return static_cast<int64_t>(v);
    return dflt;
  }
  return dflt;
}

float GGufModel::meta_float(const std::string& key, float dflt) const {
  for (const auto& kv : kv_) {
    if (kv.first != key) continue;
    char* endp = nullptr;
    const float v = std::strtof(kv.second.c_str(), &endp);
    if (endp != kv.second.c_str()) return v;
    return dflt;
  }
  return dflt;
}

}  // namespace presto::own

// ---------------------------------------------------------------------------
// self-test CLI (only with OWN_GGUF_SELFTEST defined)
// ---------------------------------------------------------------------------
#ifdef OWN_GGUF_SELFTEST
int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argc > 0 ? argv[0] : "selftest");
    return 2;
  }
  presto::own::GGufModel m;
  std::string err;
  if (!presto::own::GGufModel::load(argv[1], m, err)) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("arch=%s n_vocab=%d n_embd=%d n_layer=%d n_head=%d n_head_kv=%d\n",
              m.arch.c_str(), m.n_vocab, m.n_embd, m.n_layer, m.n_head,
              m.n_head_kv);
  std::printf("n_tensors=%zu\n", presto::own::g_selftest_names.size());
  for (std::size_t i = 0; i < presto::own::g_selftest_names.size() && i < 3; ++i)
    std::printf("  [%zu] %s\n", i, presto::own::g_selftest_names[i].c_str());

  if (presto::own::g_selftest_names.size() <= 8) {
    for (const std::string& name : presto::own::g_selftest_names) {
      const presto::own::Tensor* t = m.tensor(name);
      if (!t || t->data.empty()) continue;
      std::printf("tensor '%s' ne=[", t->name.c_str());
      for (uint32_t d : t->ne) std::printf("%u ", d);
      std::printf("] head:");
      for (std::size_t i = 0; i < t->data.size() && i < 8; ++i)
        std::printf(" %.8g", static_cast<double>(t->data[i]));
      std::printf("\n");
    }
  }

  // tokenizer round-trip smoke test
  const std::vector<int> ids = m.encode("Once upon a time");
  std::printf("encode(\"Once upon a time\") -> %zu ids:", ids.size());
  for (int id : ids) std::printf(" %d", id);
  std::printf("\ndecode -> \"%s\"\n", m.decode(ids).c_str());
  return 0;
}
#endif  // OWN_GGUF_SELFTEST
