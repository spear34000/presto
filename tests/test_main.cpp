// presto tests - entry point
#include "test_util.hpp"

#include <cstdio>

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  return presto::testing::run_all(argc > 0 ? argv[0] : "presto_tests") ? 0 : 1;
}
