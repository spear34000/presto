// presto - GGUF header/metadata parser (pure C++, spec: ggml docs/gguf.md)
#include "presto/gguf_meta.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>

namespace presto {
namespace {

constexpr std::size_t kMaxStoredString = 256 * 1024;  // cap per stored value

enum GGUFValueType : std::uint32_t {
  kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3,
  kUint32 = 4, kInt32 = 5, kFloat32 = 6, kBool = 7,
  kString = 8, kArray = 9, kUint64 = 10, kInt64 = 11, kFloat64 = 12,
};

template <typename T>
T read_le(const unsigned char* b) {
  static_assert(std::is_integral_v<T>, "read_le primary template is integral-only");
  T v{};
  for (std::size_t i = 0; i < sizeof(T); ++i)
    v |= static_cast<T>(b[i]) << (8 * i);
  return v;
}

template <>
inline float read_le<float>(const unsigned char* b) {
  const std::uint32_t bits = read_le<std::uint32_t>(b);
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

template <>
inline double read_le<double>(const unsigned char* b) {
  const std::uint64_t bits = read_le<std::uint64_t>(b);
  double d;
  std::memcpy(&d, &bits, sizeof d);
  return d;
}

struct Cursor {
  const unsigned char* p = nullptr;
  const unsigned char* end = nullptr;
  std::string& err;

  bool fail(const char* m) {
    if (err.empty()) err = m;
    return false;
  }
  std::size_t left() const { return static_cast<std::size_t>(end - p); }
  bool need(std::size_t n) { return left() >= n || fail("truncated metadata"); }
  bool u8(std::uint8_t& v) {
    if (!need(1)) return false;
    v = *p++;
    return true;
  }
  bool u16(std::uint16_t& v) {
    if (!need(2)) return false;
    v = read_le<std::uint16_t>(p);
    p += 2;
    return true;
  }
  bool u32(std::uint32_t& v) {
    if (!need(4)) return false;
    v = read_le<std::uint32_t>(p);
    p += 4;
    return true;
  }
  bool u64(std::uint64_t& v) {
    if (!need(8)) return false;
    v = read_le<std::uint64_t>(p);
    p += 8;
    return true;
  }

  template <typename T>
  bool pod(T& v) {  // little-endian fixed-size numeric
    static_assert(std::is_trivially_copyable_v<T>);
    if (!need(sizeof(T))) return false;
    v = read_le<T>(p);
    p += sizeof(T);
    return true;
  }
};

// GGUF string: u64 length (v1: u32) + bytes. Always advances cursor correctly.
bool read_gguf_string(Cursor& c, bool v1, std::string& out) {
  std::uint64_t len = 0;
  if (v1) {
    std::uint32_t l32 = 0;
    if (!c.u32(l32)) return false;
    len = l32;
  } else {
    if (!c.u64(len)) return false;
  }
  if (c.left() < len) return c.fail("truncated string");
  if (len <= kMaxStoredString)
    out.assign(reinterpret_cast<const char*>(c.p), static_cast<std::size_t>(len));
  c.p += len;
  return true;
}

std::size_t scalar_size(std::uint32_t t) {
  switch (t) {
    case kUint8:
    case kInt8:
    case kBool: return 1;
    case kUint16:
    case kInt16: return 2;
    case kFloat32:
    case kUint32:
    case kInt32: return 4;
    case kFloat64:
    case kUint64:
    case kInt64: return 8;
    default: return 0;  // variable-length or invalid
  }
}

// Skip any value; arrays of scalars skip by stride, arrays of strings walk.
bool skip_value(Cursor& c, bool v1, std::uint32_t type) {
  switch (type) {
    case kUint8:
    case kInt8:
    case kBool: return c.need(1) && (c.p += 1, true);
    case kUint16:
    case kInt16: return c.need(2) && (c.p += 2, true);
    case kFloat32:
    case kUint32:
    case kInt32: return c.need(4) && (c.p += 4, true);
    case kFloat64:
    case kUint64:
    case kInt64: return c.need(8) && (c.p += 8, true);
    case kString: {
      std::string tmp;
      return read_gguf_string(c, v1, tmp);
    }
    case kArray: {
      std::uint32_t etype = 0;
      std::uint64_t count = 0;
      if (!c.u32(etype)) return false;
      if (v1) {
        std::uint32_t n32 = 0;
        if (!c.u32(n32)) return false;
        count = n32;
      } else {
        if (!c.u64(count)) return false;
      }
      if (etype == kArray) return c.fail("nested array not allowed by spec");
      const std::size_t esz = scalar_size(etype);
      if (esz > 0) {
        // overflow-safe stride check
        if (count != 0 &&
            static_cast<std::uint64_t>(esz) * count > static_cast<std::uint64_t>(c.left()))
          return c.fail("array longer than remaining metadata");
        c.p += static_cast<std::size_t>(count) * esz;
        return true;
      }
      if (etype != kString) return c.fail("bad array element type");
      for (std::uint64_t i = 0; i < count; ++i) {
        std::string tmp;
        if (!read_gguf_string(c, v1, tmp)) return false;
      }
      return true;
    }
    default: return c.fail("unknown value type");
  }
}

bool store_scalar(Cursor& c, std::uint32_t type, std::string& s) {
  std::ostringstream o;
  switch (type) {
    case kUint8: {
      std::uint8_t v = 0;
      if (!c.u8(v)) return false;
      o << static_cast<unsigned>(v);
      break;
    }
    case kInt8: {
      std::int8_t v = 0;
      if (!c.u8(reinterpret_cast<std::uint8_t&>(v))) return false;
      o << static_cast<int>(v);
      break;
    }
    case kBool: {
      std::uint8_t v = 0;
      if (!c.u8(v)) return false;
      o << (v ? "true" : "false");
      break;
    }
    case kUint16: {
      std::uint16_t v = 0;
      if (!c.u16(v)) return false;
      o << v;
      break;
    }
    case kInt16: {
      std::int16_t v = 0;
      if (!c.u16(reinterpret_cast<std::uint16_t&>(v))) return false;
      o << static_cast<int>(v);
      break;
    }
    case kUint32: {
      std::uint32_t v = 0;
      if (!c.u32(v)) return false;
      o << v;
      break;
    }
    case kInt32: {
      std::int32_t v = 0;
      if (!c.u32(reinterpret_cast<std::uint32_t&>(v))) return false;
      o << v;
      break;
    }
    case kUint64: {
      std::uint64_t v = 0;
      if (!c.u64(v)) return false;
      o << v;
      break;
    }
    case kInt64: {
      std::int64_t v = 0;
      if (!c.u64(reinterpret_cast<std::uint64_t&>(v))) return false;
      o << v;
      break;
    }
    case kFloat32: {
      float v = 0.f;
      if (!c.pod<float>(v)) return false;
      o << v;
      break;
    }
    case kFloat64: {
      double v = 0.0;
      if (!c.pod<double>(v)) return false;
      o << v;
      break;
    }
    default: return c.fail("not a scalar");
  }
  s = o.str();
  return true;
}

} // namespace

const char* ggml_ftype_name(std::int64_t ft) {
  switch (ft) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 6: return "Q5_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 9: return "Q8_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "IQ2_XXS";
    case 16: return "IQ2_XS";
    default: return nullptr;
  }
}

std::map<std::string, std::string> GgufInfo::to_meta() const {
  std::map<std::string, std::string> m;
  m["gguf_version"] = std::to_string(version);
  m["tensor_count"] = std::to_string(tensor_count);
  m["kv_count"] = std::to_string(kv_count);
  if (!architecture.empty()) m["arch"] = architecture;
  if (!name.empty()) m["name"] = name;
  if (file_type >= 0) {
    if (const char* qn = ggml_ftype_name(file_type))
      m["quant"] = qn;
    else
      m["quant"] = "ftype_" + std::to_string(file_type);
  }
  return m;
}

bool parse_gguf_header(const std::string& path, GgufInfo& out, std::string& err) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    err = "cannot open file";
    return false;
  }
  const std::streamoff fsize = f.tellg();
  f.seekg(0);

  constexpr std::streamoff kMaxMetadataBytes = 64 * 1024 * 1024;  // pathological-file guard
  const std::streamoff want = std::min<std::streamoff>(fsize, kMaxMetadataBytes);
  std::vector<unsigned char> buf(static_cast<std::size_t>(want));
  f.read(reinterpret_cast<char*>(buf.data()), want);
  const auto got = static_cast<std::size_t>(f.gcount());

  Cursor c{buf.data(), buf.data() + got, err};
  if (!c.need(4)) return false;
  if (!(c.p[0] == 'G' && c.p[1] == 'G' && c.p[2] == 'U' && c.p[3] == 'F')) {
    err = "missing GGUF magic";
    return false;
  }
  c.p += 4;

  if (!c.u32(out.version)) return false;
  if (out.version < 1 || out.version > 3) {
    err = "unsupported GGUF version " + std::to_string(out.version);
    return false;
  }
  const bool v1 = out.version == 1;

  if (v1) {
    std::uint32_t t32 = 0, k32 = 0;
    if (!c.u32(t32) || !c.u32(k32)) return false;
    out.tensor_count = t32;
    out.kv_count = k32;
  } else {
    if (!c.u64(out.tensor_count) || !c.u64(out.kv_count)) return false;
  }

  for (std::uint64_t i = 0; i < out.kv_count; ++i) {
    std::string key;
    if (!read_gguf_string(c, v1, key)) {
      if (err.empty()) err = "failed reading key #" + std::to_string(i);
      return false;
    }
    std::uint32_t type = 0;
    if (!c.u32(type)) return false;

    if (type == kString) {
      std::string val;
      if (!read_gguf_string(c, v1, val)) return false;
      if (!val.empty()) out.kvs[key] = val;
    } else if (type == kArray) {
      if (!skip_value(c, v1, type)) return false;
      out.kvs[key] = "<array>";
    } else {
      std::string s;
      if (!store_scalar(c, type, s)) return false;
      out.kvs[key] = s;
    }
  }

  if (auto it = out.kvs.find("general.file_type"); it != out.kvs.end())
    out.file_type = std::strtoll(it->second.c_str(), nullptr, 10);
  if (auto it = out.kvs.find("general.architecture"); it != out.kvs.end())
    out.architecture = it->second;
  if (auto it = out.kvs.find("general.name"); it != out.kvs.end())
    out.name = it->second;
  return true;
}

} // namespace presto
