// presto - minimal dependency-free JSON parser (header-only)
// RFC 8259 subset: objects, arrays, strings (with \uXXXX + surrogate pairs),
// numbers (int64 / double), booleans, null. Depth-bounded, no exceptions.
#pragma once

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace presto::json {

class Node {
public:
  enum class Type : std::uint8_t { Null, Bool, Int, Double, String, ArrayT, ObjectT };

  Node() = default;

  static Node null_node() { return Node(); }
  static Node boolean(bool b) {
    Node n;
    n.type_ = Type::Bool;
    n.bool_ = b;
    return n;
  }
  static Node integer(std::int64_t v) {
    Node n;
    n.type_ = Type::Int;
    n.int_ = v;
    return n;
  }
  static Node number(double d) {
    Node n;
    n.type_ = Type::Double;
    n.dbl_ = d;
    return n;
  }
  static Node str(std::string s) {
    Node n;
    n.type_ = Type::String;
    n.str_ = std::move(s);
    return n;
  }
  static Node array() {
    Node n;
    n.type_ = Type::ArrayT;
    return n;
  }
  static Node object() {
    Node n;
    n.type_ = Type::ObjectT;
    return n;
  }

  bool is_null() const { return type_ == Type::Null; }
  bool is_bool() const { return type_ == Type::Bool; }
  bool is_int() const { return type_ == Type::Int; }
  bool is_double() const { return type_ == Type::Double; }
  bool is_number() const { return is_int() || is_double(); }
  bool is_string() const { return type_ == Type::String; }
  bool is_array() const { return type_ == Type::ArrayT; }
  bool is_object() const { return type_ == Type::ObjectT; }

  bool as_bool(bool dflt = false) const { return is_bool() ? bool_ : dflt; }
  std::int64_t as_int(std::int64_t dflt = 0) const {
    if (is_int()) return int_;
    if (is_double()) return static_cast<std::int64_t>(dbl_);
    return dflt;
  }
  double as_double(double dflt = 0.0) const {
    if (is_int()) return static_cast<double>(int_);
    if (is_double()) return dbl_;
    return dflt;
  }
  const std::string& as_string() const { return str_; }

  // array accessors
  const std::vector<Node>& items() const { return arr_; }
  void push(Node v) { arr_.push_back(std::move(v)); }

  // object accessors
  const std::map<std::string, Node>& members() const { return obj_; }
  void set(std::string k, Node v) { obj_[std::move(k)] = std::move(v); }
  const Node* find(const std::string& key) const {
    if (!is_object()) return nullptr;
    auto it = obj_.find(key);
    return it == obj_.end() ? nullptr : &it->second;
  }
  bool contains(const std::string& key) const { return find(key) != nullptr; }

private:
  Type type_ = Type::Null;
  bool bool_ = false;
  std::int64_t int_ = 0;
  double dbl_ = 0.0;
  std::string str_;
  std::vector<Node> arr_;
  std::map<std::string, Node> obj_;
};

namespace detail {

struct Parser {
  const char* p = nullptr;
  const char* end = nullptr;
  std::string err;
  int depth = 0;
  static constexpr int kMaxDepth = 128;

  Parser(const char* begin, const char* stop) : p(begin), end(stop) {}

  bool fail(const char* m) {
    if (err.empty()) err = m;
    return false;
  }

  void skip_ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }

  bool literal(const char* lit, const char* what) {
    for (; *lit; ++lit, ++p)
      if (p >= end || *p != *lit) return fail(what);
    return true;
  }

  bool value(Node& out) {
    if (++depth > kMaxDepth) return fail("nesting too deep");
    skip_ws();
    if (p >= end) return fail("unexpected end of input");
    bool ok = false;
    switch (*p) {
      case '{': ok = object(out); break;
      case '[': ok = array(out); break;
      case '"': {
        std::string s;
        if ((ok = string_lit(s))) out = Node::str(std::move(s));
        break;
      }
      case 't':
        if (!literal("true", "bad literal")) { --depth; return false; }
        out = Node::boolean(true);
        ok = true;
        break;
      case 'f':
        if (!literal("false", "bad literal")) { --depth; return false; }
        out = Node::boolean(false);
        ok = true;
        break;
      case 'n':
        if (!literal("null", "bad literal")) { --depth; return false; }
        out = Node();
        ok = true;
        break;
      default:
        if (*p == '-' || std::isdigit(static_cast<unsigned char>(*p)))
          ok = number(out);
        else
          return fail("unexpected character"), --depth, false;
        break;
    }
    --depth;
    return ok;
  }

  bool object(Node& out) {
    out = Node::object();
    ++p; // consume '{'
    skip_ws();
    if (p < end && *p == '}') {
      ++p;
      return true;
    }
    while (true) {
      skip_ws();
      std::string key;
      if (!string_lit(key)) return false;
      skip_ws();
      if (p >= end || *p != ':') return fail("expected ':'");
      ++p;
      Node val;
      if (!value(val)) return false;
      out.set(key, std::move(val));
      skip_ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == '}') {
        ++p;
        return true;
      }
      return fail("expected ',' or '}'");
    }
  }

  bool array(Node& out) {
    out = Node::array();
    ++p; // consume '['
    skip_ws();
    if (p < end && *p == ']') {
      ++p;
      return true;
    }
    while (true) {
      Node val;
      if (!value(val)) return false;
      out.push(std::move(val));
      skip_ws();
      if (p < end && *p == ',') {
        ++p;
        continue;
      }
      if (p < end && *p == ']') {
        ++p;
        return true;
      }
      return fail("expected ',' or ']'");
    }
  }

  bool number(Node& out) {
    const char* start = p;
    if (p < end && *p == '-') ++p;
    while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    bool is_float = false;
    if (p < end && *p == '.') {
      is_float = true;
      ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    }
    if (p < end && (*p == 'e' || *p == 'E')) {
      is_float = true;
      ++p;
      if (p < end && (*p == '+' || *p == '-')) ++p;
      while (p < end && std::isdigit(static_cast<unsigned char>(*p))) ++p;
    }
    const std::string tok(start, p);
    if (tok.empty() || tok == "-") return fail("bad number");
    if (!is_float) {
      errno = 0;
      char* e = nullptr;
      const long long v = std::strtoll(tok.c_str(), &e, 10);
      if (e && *e == '\0' && errno != ERANGE) {
        out = Node::integer(v);
        return true;
      }
    }
    out = Node::number(std::strtod(tok.c_str(), nullptr));
    return true;
  }

  bool string_lit(std::string& s) {
    if (p >= end || *p != '"') return fail("expected string");
    ++p;
    s.clear();
    while (true) {
      if (p >= end) return fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(*p);
      if (c == '"') {
        ++p;
        return true;
      }
      if (c == '\\') {
        ++p;
        if (p >= end) return fail("bad escape");
        const char e = *p++;
        switch (e) {
          case '"': s.push_back('"'); break;
          case '\\': s.push_back('\\'); break;
          case '/': s.push_back('/'); break;
          case 'b': s.push_back('\b'); break;
          case 'f': s.push_back('\f'); break;
          case 'n': s.push_back('\n'); break;
          case 'r': s.push_back('\r'); break;
          case 't': s.push_back('\t'); break;
          case 'u': {
            unsigned cp = 0;
            if (!hex4(cp)) return false;
            if (cp >= 0xD800u && cp <= 0xDBFFu && p + 1 < end && p[0] == '\\' && p[1] == 'u') {
              const char* save = p;
              p += 2;
              unsigned lo = 0;
              if (!hex4(lo)) {
                p = save; // treat lone surrogate literally
              } else if (lo >= 0xDC00u && lo <= 0xDFFFu) {
                cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
              } else {
                p = save;
              }
            }
            utf8_append(s, cp);
            break;
          }
          default: return fail("unknown escape");
        }
      } else {
        s.push_back(static_cast<char>(c));
        ++p;
      }
    }
  }

  bool hex4(unsigned& v) {
    v = 0;
    for (int i = 0; i < 4; ++i) {
      if (p >= end) return fail("bad \\u escape");
      const char c = *p++;
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f')
        v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        v |= static_cast<unsigned>(c - 'A' + 10);
      else
        return fail("bad hex digit");
    }
    return true;
  }

  static void utf8_append(std::string& s, unsigned cp) {
    if (cp < 0x80u) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800u) {
      s.push_back(static_cast<char>(0xC0u | (cp >> 6)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp < 0x10000u) {
      s.push_back(static_cast<char>(0xE0u | (cp >> 12)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
      s.push_back(static_cast<char>(0xF0u | (cp >> 18)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
      s.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
  }
};

} // namespace detail

// Parse a complete JSON document. On failure returns false and fills err.
inline bool parse(const std::string& text, Node& out, std::string& err) {
  detail::Parser ps(text.data(), text.data() + text.size());
  if (!ps.value(out)) {
    err = ps.err.empty() ? "parse failed" : ps.err;
    return false;
  }
  ps.skip_ws();
  if (ps.p != ps.end) {
    err = "trailing characters after JSON value";
    return false;
  }
  return true;
}

} // namespace presto::json
