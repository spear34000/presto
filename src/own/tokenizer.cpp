// presto own-engine: SentencePiece-style BPE tokenizer over GGUF-embedded
// vocab (tokenizer.ggml.* metadata arrays). Pure C++20, no sentencepiece /
// llama.cpp dependency.
//
// encode(): SPM conventions - prepend one space, map ' ' -> U+2581 (▁),
// split into UTF-8 chunks, unknown chars fall back to <0xNN> byte pieces,
// then repeatedly merge the adjacent pair whose concatenation exists in the
// vocab with the highest score until no merge is possible. BOS id is
// prepended when add_bos is set (from tokenizer.ggml.bos_token_id).
//
// decode(): concat pieces, convert <0xNN> byte-fallback pieces back to raw
// bytes, ▁ -> ' ', skip control/unknown/unused tokens, strip the single
// leading space per SPM convention.
//
// The GGufModel contract fixes the private member set (no vocab cache), so
// both encode() and decode() re-walk the metadata section of buf_ (the whole
// file image) to rebuild the vocab. That walk touches only the kv section -
// microseconds for a 32k SPM vocab - and keeps the class stateless.
#include "own.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace presto::own {
namespace {

// ---------------------------------------------------------------------------
// minimal GGUF metadata walker (mirrors gguf_load.cpp; kept independent so
// each translation unit stands alone)
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
};

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
    default: return 0;
  }
}

bool skip_value(Rd& r, bool narrow_counts, uint32_t t) {
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
  if (et == kV_ARR) {
    r.bad = true;
    return false;
  }
  const uint32_t w = scalar_width(et);
  if (w != 0) {
    if (n > r.left() / w) {
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

// ---------------------------------------------------------------------------
// vocab extracted from tokenizer.ggml.* arrays
// ---------------------------------------------------------------------------
struct Vocab {
  std::unordered_map<std::string, int> piece_to_id;
  std::vector<std::string> pieces;
  std::vector<float> scores;   // empty -> all merges score 0
  std::vector<uint8_t> ttype;  // gguf token_type per id; empty -> unknown
  int bos = -1;
};

// gguf token_type values
constexpr uint8_t kTtUnknown = 2, kTtControl = 3, kTtUnused = 5;

bool extract_vocab(const std::vector<char>& buf, Vocab& v, std::string& err) {
  if (buf.size() < 24) {
    err = "model buffer too small";
    return false;
  }
  Rd r{reinterpret_cast<const unsigned char*>(buf.data()),
       reinterpret_cast<const unsigned char*>(buf.data()) + buf.size(), false};
  if (!r.need(4) || std::memcmp(r.p, "GGUF", 4) != 0) {
    err = "bad GGUF magic";
    return false;
  }
  r.p += 4;
  const uint32_t version = r.u32();
  if (r.bad || version < 1 || version > 3) {
    err = "unsupported GGUF version";
    return false;
  }
  const bool narrow_counts = (version == 1);
  const uint64_t n_tensors = narrow_counts ? r.u32() : r.u64();  // unused here
  const uint64_t n_kv = narrow_counts ? r.u32() : r.u64();
  if (r.bad || n_tensors > (1ull << 20) || n_kv > (1ull << 20)) {
    err = "corrupt GGUF header";
    return false;
  }

  bool have_tokens = false, have_scores = false, have_ttype = false,
       have_bos = false;
  for (uint64_t i = 0; i < n_kv && !(have_tokens && have_scores && have_ttype && have_bos);
       ++i) {
    std::string key;
    if (!rd_string(r, narrow_counts, key)) {
      err = "truncated metadata";
      return false;
    }
    const uint32_t vt = r.u32();
    if (r.bad) {
      err = "truncated metadata";
      return false;
    }

    if (vt == kV_ARR && key == "tokenizer.ggml.tokens") {
      const uint32_t et = r.u32();
      const uint64_t n = narrow_counts ? r.u32() : r.u64();
      if (r.bad || et != kV_STR || n > (1ull << 21)) {
        err = "tokenizer.ggml.tokens: expected string array";
        return false;
      }
      v.pieces.resize(static_cast<std::size_t>(n));
      for (uint64_t j = 0; j < n; ++j)
        if (!rd_string(r, narrow_counts, v.pieces[static_cast<std::size_t>(j)])) {
          err = "truncated tokenizer.ggml.tokens";
          return false;
        }
      have_tokens = true;
    } else if (vt == kV_ARR && key == "tokenizer.ggml.scores") {
      const uint32_t et = r.u32();
      const uint64_t n = narrow_counts ? r.u32() : r.u64();
      if (r.bad || et != kV_F32 || n > (1ull << 21)) {
        err = "tokenizer.ggml.scores: expected f32 array";
        return false;
      }
      v.scores.resize(static_cast<std::size_t>(n));
      for (uint64_t j = 0; j < n; ++j) {
        // little-endian f32
        if (!r.need(4)) {
          err = "truncated tokenizer.ggml.scores";
          return false;
        }
        const uint32_t bits = r.u32();
        std::memcpy(&v.scores[static_cast<std::size_t>(j)], &bits, 4);
      }
      have_scores = true;
    } else if (vt == kV_ARR && key == "tokenizer.ggml.token_type") {
      const uint32_t et = r.u32();
      const uint64_t n = narrow_counts ? r.u32() : r.u64();
      if (r.bad || et != kV_I32 || n > (1ull << 21)) {
        err = "tokenizer.ggml.token_type: expected i32 array";
        return false;
      }
      v.ttype.resize(static_cast<std::size_t>(n));
      for (uint64_t j = 0; j < n; ++j) {
        const uint32_t x = r.u32();
        if (r.bad) {
          err = "truncated tokenizer.ggml.token_type";
          return false;
        }
        v.ttype[static_cast<std::size_t>(j)] =
            static_cast<uint8_t>(static_cast<int32_t>(x) & 0xFF);
      }
      have_ttype = true;
    } else if ((vt == kV_I32 || vt == kV_U32) &&
               key == "tokenizer.ggml.bos_token_id") {
      v.bos = static_cast<int32_t>(r.u32());
      if (r.bad) {
        err = "truncated bos_token_id";
        return false;
      }
      have_bos = true;
    } else if (vt == kV_STR) {
      std::string tmp;
      if (!rd_string(r, narrow_counts, tmp)) {
        err = "truncated metadata string at '" + key + "'";
        return false;
      }
    } else if (!skip_value(r, narrow_counts, vt)) {
      err = "malformed metadata at '" + key + "'";
      return false;
    }
  }

  if (!have_tokens) {
    err = "model has no tokenizer.ggml.tokens array";
    return false;
  }
  if (!have_scores) v.scores.assign(v.pieces.size(), 0.f);

  v.piece_to_id.reserve(v.pieces.size() * 2);
  for (std::size_t id = 0; id < v.pieces.size(); ++id)
    v.piece_to_id.emplace(v.pieces[id], static_cast<int>(id));
  return true;
}

float score_of(const Vocab& v, int id) {
  if (id >= 0 && static_cast<std::size_t>(id) < v.scores.size())
    return v.scores[static_cast<std::size_t>(id)];
  return 0.f;
}

// UTF-8 sequence length from lead byte; 1 for invalid leads (byte fallback).
std::size_t utf8_len(unsigned char c) {
  if ((c & 0x80u) == 0) return 1;
  if ((c & 0xE0u) == 0xC0) return 2;
  if ((c & 0xF0u) == 0xE0) return 3;
  if ((c & 0xF8u) == 0xF0) return 4;
  return 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
std::vector<int> GGufModel::encode(const std::string& text) const {
  std::vector<int> ids;
  Vocab v;
  std::string err;
  if (!extract_vocab(buf_, v, err)) return ids;  // no vocab -> no tokens

  if (add_bos && v.bos >= 0 &&
      static_cast<std::size_t>(v.bos) < v.pieces.size())
    ids.push_back(v.bos);
  if (text.empty()) return ids;

  // SPM normalization: prepend one space, then ' ' -> U+2581 (▁).
  std::string norm = " ";
  norm += text;
  const char shaq[4] = {'\xe2', '\x96', '\x81', '\0'};
  for (std::size_t i = 0; i < norm.size();) {
    if (norm[i] == ' ') {
      norm.replace(i, 1, shaq);
      i += 3;
    } else {
      ++i;
    }
  }

  // Split into UTF-8 chunks; initial pieces are vocab hits or <0xNN> bytes.
  std::vector<std::string> pieces;
  pieces.reserve(norm.size());
  for (std::size_t i = 0; i < norm.size();) {
    std::size_t len = utf8_len(static_cast<unsigned char>(norm[i]));
    if (i + len > norm.size()) len = 1;  // truncated sequence -> byte fallback
    const std::string chunk = norm.substr(i, len);
    i += len;
    if (v.piece_to_id.find(chunk) != v.piece_to_id.end()) {
      pieces.push_back(chunk);
    } else {
      char bp[8];
      for (char ch : chunk) {
        std::snprintf(bp, sizeof bp, "<0x%02X>",
                      static_cast<unsigned>(static_cast<unsigned char>(ch)));
        pieces.emplace_back(bp);
      }
    }
  }

  // Merge loop: repeatedly merge the adjacent pair whose concatenation is in
  // the vocab and has the highest score (leftmost wins ties). Deterministic.
  while (pieces.size() > 1) {
    std::size_t best = pieces.size();
    float best_score = -1e30f;
    for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
      const auto it = v.piece_to_id.find(pieces[i] + pieces[i + 1]);
      if (it == v.piece_to_id.end()) continue;
      const float s = score_of(v, it->second);
      if (best == pieces.size() || s > best_score) {
        best = i;
        best_score = s;
      }
    }
    if (best == pieces.size()) break;
    pieces[best] += pieces[best + 1];
    pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best) + 1);
  }

  for (const std::string& p : pieces) {
    const auto it = v.piece_to_id.find(p);
    if (it != v.piece_to_id.end()) ids.push_back(it->second);
    // piece not in vocab (e.g. model without byte fallback): dropped
  }
  return ids;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
std::string GGufModel::decode(const std::vector<int>& token_ids) const {
  Vocab v;
  std::string err;
  if (!extract_vocab(buf_, v, err)) return "";

  auto hexval = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };

  std::string out;
  out.reserve(token_ids.size() * 4);
  for (int id : token_ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= v.pieces.size()) continue;
    if (!v.ttype.empty() && static_cast<std::size_t>(id) < v.ttype.size()) {
      const uint8_t tt = v.ttype[static_cast<std::size_t>(id)];
      if (tt == kTtUnknown || tt == kTtControl || tt == kTtUnused) continue;
    }
    const std::string& s = v.pieces[static_cast<std::size_t>(id)];

    // byte-fallback piece "<0xNN>" -> raw byte
    if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' &&
        s[5] == '>') {
      const int hi = hexval(s[3]);
      const int lo = hexval(s[4]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>(hi * 16 + lo));
        continue;
      }
    }

    // ▁ (U+2581) -> ' '
    for (std::size_t i = 0; i < s.size();) {
      if (i + 3 <= s.size() && s[i] == '\xe2' && s[i + 1] == '\x96' &&
          s[i + 2] == '\x81') {
        out.push_back(' ');
        i += 3;
      } else {
        out.push_back(s[i]);
        ++i;
      }
    }
  }

  // SPM convention: strip the single leading space added by the normalizer.
  if (!out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

}  // namespace presto::own
