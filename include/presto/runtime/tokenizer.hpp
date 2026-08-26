#pragma once

#include "presto/runtime/status.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace presto::runtime {

enum class TokenizerModel { Bpe, Unigram };

struct TokenizeOptions {
  bool add_bos = false;
  bool add_eos = false;
  bool parse_special = true;
};

struct TokenizerConfig {
  TokenizerModel model = TokenizerModel::Bpe;
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  std::vector<double> scores;
  std::vector<std::int32_t> token_types;
  std::int32_t bos_id = -1;
  std::int32_t eos_id = -1;
  std::int32_t unk_id = -1;
  bool add_bos = false;
  bool add_eos = false;
  std::string pre_tokenizer;
};

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  virtual Result<std::vector<std::int32_t>> encode(
      std::string_view text, TokenizeOptions options = {}) const = 0;
  virtual Result<std::string> decode(
      const std::vector<std::int32_t>& ids) const = 0;
};

Result<std::unique_ptr<Tokenizer>> make_tokenizer(const TokenizerConfig& config);

} // namespace presto::runtime
