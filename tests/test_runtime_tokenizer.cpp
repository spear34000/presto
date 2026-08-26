#include "presto/runtime/tokenizer.hpp"
#include "test_util.hpp"

#include <vector>

using namespace presto::runtime;

PRESTO_TEST(runtime_bpe_merges_gpt2_bytes_and_round_trips) {
  TokenizerConfig config;
  config.tokens = {"<unk>", "H", "e", "l", "o", "Ġ", "w", "r", "d",
                   "Hello", "Ġworld"};
  config.merges = {"H e", "He l", "Hel l", "Hell o", "Ġ w", "Ġw o",
                   "Ġwo r", "Ġwor l", "Ġworl d"};
  config.unk_id = 0;
  const auto tokenizer = make_tokenizer(config);
  PRESTO_EXPECT(tokenizer.ok());
  if (!tokenizer.ok()) return;
  const auto ids = tokenizer.value->encode("Hello world");
  PRESTO_EXPECT(ids.ok());
  if (!ids.ok()) return;
  PRESTO_EXPECT(ids.value == std::vector<std::int32_t>({9, 10}));
  const auto text = tokenizer.value->decode(ids.value);
  PRESTO_EXPECT(text.ok() && text.value == "Hello world");
}

PRESTO_TEST(runtime_bpe_handles_utf8_and_special_tokens) {
  TokenizerConfig config;
  config.tokens = {"<unk>", "<|im_start|>", "한국어"};
  config.unk_id = 0;
  const auto tokenizer = make_tokenizer(config);
  PRESTO_EXPECT(tokenizer.ok());
  if (!tokenizer.ok()) return;
  const auto ids = tokenizer.value->encode("<|im_start|>한국어");
  PRESTO_EXPECT(ids.ok());
  if (!ids.ok()) return;
  PRESTO_EXPECT(ids.value == std::vector<std::int32_t>({1, 2}));
  const auto text = tokenizer.value->decode(ids.value);
  PRESTO_EXPECT(text.ok() && text.value == "<|im_start|>한국어");
}

PRESTO_TEST(runtime_bpe_rejects_malformed_configuration) {
  TokenizerConfig config;
  config.tokens = {"<unk>", "a"};
  config.merges = {"a"};
  PRESTO_EXPECT(!make_tokenizer(config).ok());
}
