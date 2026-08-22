// presto - tiny zero-dependency test harness (shared by all test TUs)
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace presto::testing {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& failure_count() {
  static int n = 0;
  return n;
}

inline bool run_all(const char* argv0) {
  (void)argv0;
  for (const auto& t : registry()) {
    const int before = failure_count();
    std::printf("[ RUN      ] %s\n", t.name);
    t.fn();
    std::printf("%s %s\n", before == failure_count() ? "[       OK ]" : "[  FAILED  ]",
                t.name);
  }
  if (failure_count() == 0) {
    std::printf("ALL TESTS PASSED (%zu)\n", registry().size());
    return true;
  }
  std::printf("TESTS FAILED: %d of %zu cases had failures\n", failure_count(),
              registry().size());
  return false;
}

} // namespace presto::testing

#define PRESTO_TEST(name)                                                        \
  static void name();                                                            \
  namespace {                                                                    \
  struct Register_##name {                                                       \
    Register_##name() { presto::testing::registry().push_back({#name, &name}); } \
  } register_##name##_instance_;                                                 \
  }                                                                              \
  static void name()

#define PRESTO_EXPECT(cond)                                                      \
  do {                                                                           \
    if (!(cond)) {                                                               \
      ++::presto::testing::failure_count();                                      \
      std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
    }                                                                            \
  } while (0)

#define PRESTO_EXPECT_STR_EQ(a, b)                                               \
  do {                                                                           \
    const std::string sa_ = (a);                                                 \
    const std::string sb_ = (b);                                                 \
    if (sa_ != sb_) {                                                            \
      ++::presto::testing::failure_count();                                      \
      std::fprintf(stderr, "  FAIL %s:%d: \"%s\" != \"%s\"\n", __FILE__,         \
                   __LINE__, sa_.c_str(), sb_.c_str());                          \
    }                                                                            \
  } while (0)
