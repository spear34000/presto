// UTF-8 boundary helpers for command-line applications.
#pragma once

#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <codecvt>
#include <locale>
#endif

namespace presto {

inline std::string utf8_from_wide(const std::wstring& value) {
#if defined(_WIN32)
  if (value.empty()) return {};
  const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (bytes <= 0) return {};
  std::string result(static_cast<size_t>(bytes), '\0');
  WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                      static_cast<int>(value.size()), result.data(), bytes, nullptr, nullptr);
  return result;
#else
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>{}.to_bytes(value);
#endif
}

} // namespace presto
