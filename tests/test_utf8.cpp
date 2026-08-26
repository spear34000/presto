#include "presto/utf8.hpp"
#include "test_util.hpp"

PRESTO_TEST(windows_command_arguments_become_utf8) {
  // This catches a regression where Windows argv reaches llama_tokenize() in
  // the active ANSI code page instead of the UTF-8 bytes the tokenizer needs.
  const std::wstring korean = L"\uD55C\uAD6D\uC5B4\uB85C \uC790\uAE30\uC18C\uAC1C\uD574\uC918";
  const unsigned char bytes[] = {
      0xED, 0x95, 0x9C, 0xEA, 0xB5, 0xAD, 0xEC, 0x96, 0xB4, 0xEB, 0xA1, 0x9C,
      0x20, 0xEC, 0x9E, 0x90, 0xEA, 0xB8, 0xB0, 0xEC, 0x86, 0x8C, 0xEA, 0xB0,
      0x9C, 0xED, 0x95, 0xB4, 0xEC, 0xA4, 0x98};
  std::string expected;
  for (const unsigned char byte : bytes) expected.push_back(static_cast<char>(byte));
  PRESTO_EXPECT_STR_EQ(presto::utf8_from_wide(korean), expected);
}
