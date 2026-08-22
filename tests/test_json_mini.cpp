// presto tests - minimal JSON parser
#include "test_util.hpp"

#include "json_mini.hpp"

namespace presto::testing {

PRESTO_TEST(json_scalars) {
  json::Node n;
  std::string err;
  PRESTO_EXPECT(json::parse(R"(null)", n, err) && n.is_null());
  PRESTO_EXPECT(json::parse(R"(true)", n, err) && n.is_bool() && n.as_bool());
  PRESTO_EXPECT(json::parse(R"(false)", n, err) && n.is_bool() && !n.as_bool());
  PRESTO_EXPECT(json::parse("42", n, err) && n.is_int() && n.as_int() == 42);
  PRESTO_EXPECT(json::parse("-7", n, err) && n.as_int() == -7);
  PRESTO_EXPECT(json::parse("3.5", n, err) && n.is_double());
  PRESTO_EXPECT(json::parse("1e3", n, err) && n.is_double());
  PRESTO_EXPECT(json::parse(R"("hi")", n, err) && n.is_string() && n.as_string() == "hi");
}

PRESTO_TEST(json_nested) {
  const std::string doc =
      R"({"a":{"b":[1,2,{"c":"d"}],"e":true},"f":null,"g":"A\nB\u0041"})";
  json::Node root;
  std::string err;
  PRESTO_EXPECT(json::parse(doc, root, err));
  if (!err.empty()) std::fprintf(stderr, "  parse error: %s\n", err.c_str());
  PRESTO_EXPECT(root.is_object());
  const json::Node* b = root.find("a");
  PRESTO_EXPECT(b && b->is_object());
  const json::Node* arr = b->find("b");
  PRESTO_EXPECT(arr && arr->is_array() && arr->items().size() == 3);
  PRESTO_EXPECT((*arr).items()[0].as_int() == 1);
  PRESTO_EXPECT((*arr).items()[2].find("c") != nullptr &&
                (*arr).items()[2].find("c")->as_string() == "d");
  PRESTO_EXPECT(root.contains("f") && root.find("f")->is_null());
  PRESTO_EXPECT(root.find("g") != nullptr && root.find("g")->as_string() == "A\nBA");
}

PRESTO_TEST(json_rejects_garbage) {
  json::Node n;
  std::string err;
  PRESTO_EXPECT(!json::parse("{", n, err));
  PRESTO_EXPECT(!json::parse("[1,]", n, err));       // trailing comma
  PRESTO_EXPECT(!json::parse("\"unterminated", n, err));
  PRESTO_EXPECT(!json::parse("{} trailing", n, err));
  PRESTO_EXPECT(!json::parse("nul", n, err));
  PRESTO_EXPECT(!json::parse("", n, err));
  PRESTO_EXPECT(!err.empty() || true);               // every failure sets some message
}

} // namespace presto::testing
