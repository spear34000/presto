// presto tests - entry point
#include "test_util.hpp"

int main(int argc, char** argv) {
  return presto::testing::run_all(argc > 0 ? argv[0] : "presto_tests") ? 0 : 1;
}
