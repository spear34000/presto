// presto - OpenAI-compatible HTTP server (cpp-httplib based)
#include "presto/server.hpp"

#include "json_mini.hpp"
#include "presto/backend.hpp"
#include "presto/engine.hpp"
#include "presto/log.hpp"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace presto {
namespace {

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (const char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

std::string timestamp_seconds() {
  return std::to_string(
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
}

struct RequestOpts {
  std::string prompt_text;
  std::vector<int> prompt_tokens;
  int max_tokens = 16;
  float temp = 0.0f;
};

bool parse_request_body(const std::string& body, const bool chat, RequestOpts& opts,
                        std::string& err) {
  json::Node root;
  if (!json::parse(body, root, err) || !root.is_object()) {
    err = "request body is not valid JSON";
    return false;
  }
  if (chat) {
    const json::Node* msgs = root.find("messages");
    if (!msgs || !msgs->is_array() || msgs->items().empty()) {
      err = "missing messages[]";
      return false;
    }
    // naive template: "<role>: <content>\n" per message (roadmap: real chat templates)
    for (const auto& m : msgs->items()) {
      const json::Node* role = m.find("role");
      const json::Node* content = m.find("content");
      if (role && role->is_string()) opts.prompt_text += role->as_string() + ": ";
      if (content && content->is_string()) opts.prompt_text += content->as_string();
      opts.prompt_text += "\n";
    }
  } else {
    const json::Node* p = root.find("prompt");
    if (p && p->is_string()) {
      opts.prompt_text = p->as_string();
    } else if (p && p->is_array()) {
      for (const auto& item : p->items())
        if (item.is_int()) opts.prompt_tokens.push_back(static_cast<int>(item.as_int()));
      if (opts.prompt_tokens.empty()) {
        err = "prompt array contained no integers";
        return false;
      }
    } else {
      err = "missing prompt";
      return false;
    }
  }
  if (const json::Node* n = root.find("max_tokens"); n && n->is_number())
    opts.max_tokens = static_cast<int>(n->as_int());
  if (opts.max_tokens <= 0 || opts.max_tokens > 4096) opts.max_tokens = 16;
  if (const json::Node* n = root.find("temperature"); n && n->is_number())
    opts.temp = static_cast<float>(n->as_double());
  return true;
}

} // namespace

int run_openai_server(const Detection& d, const std::string& host, int port) {
  std::string err;
  auto backend = select_backend(d, err);
  if (!backend) {
    PRESTO_LOG_ERROR("serve", err);
    return backend_caps().llamacpp || backend_caps().mlx ? 5 : 4;
  }
  if (!backend->load(err)) {
    PRESTO_LOG_ERROR("serve", "model load failed: " + err);
    return 5;
  }

  httplib::Server svr;
  std::atomic<long long> req_counter{0};
  // one engine instance -> generation must be serialized; httplib dispatches
  // connections on multiple threads
  std::mutex gen_mutex;
  const std::string model_field = json_escape(d.path);

  svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok","backend":")" + std::string(backend->name()) +
                        R"(","format":")" + format_name(d.format) + R"("})",
                    "application/json");
  });

  svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"object":"list","data":[{"id":")" + model_field +
                        R"(","object":"model","owned_by":"presto"}]})",
                    "application/json");
  });

  auto handle_generate = [&](bool chat, const httplib::Request& req,
                             httplib::Response& res) {
    std::lock_guard<std::mutex> lock(gen_mutex);
    RequestOpts opts;
    std::string perr;
    if (!parse_request_body(req.body, chat, opts, perr)) {
      res.status = 400;
      res.set_content(R"({"error":{"message":")" + json_escape(perr) + R"("}})",
                      "application/json");
      return;
    }
    GenerateParams gp;
    gp.prompt_text = opts.prompt_text;
    gp.prompt_tokens = opts.prompt_tokens;
    gp.max_tokens = opts.max_tokens;
    gp.temp = opts.temp;

    GenerateResult r;
    if (!backend->generate(gp, r, err)) {
      PRESTO_LOG_ERROR("serve", "generate failed: " + err);
      res.status = 500;
      res.set_content(R"({"error":{"message":")" + json_escape(err) + R"("}})",
                      "application/json");
      return;
    }

    const long long id = ++req_counter;
    const std::string created = timestamp_seconds();
    const std::string text = json_escape(r.text);
    if (chat) {
      res.set_content(R"({"id":"chatcmpl-presto-)" + std::to_string(id) +
                          R"(","object":"chat.completion","created":)" + created +
                          R"(,"model":")" + model_field +
                          R"(","choices":[{"index":0,"message":{"role":"assistant","content":")" +
                          text + R"("},"finish_reason":"stop"}],"usage":{"completion_tokens":)" +
                          std::to_string(r.tokens.size()) + R"(}})",
                      "application/json");
    } else {
      res.set_content(R"({"id":"cmpl-presto-)" + std::to_string(id) +
                          R"(","object":"text_completion","created":)" + created +
                          R"(,"model":")" + model_field +
                          R"(","choices":[{"index":0,"text":")" + text +
                          R"(","finish_reason":"stop"}],"usage":{"completion_tokens":)" +
                          std::to_string(r.tokens.size()) + R"(}})",
                      "application/json");
    }
  };

  svr.Post("/v1/completions",
           [&](const httplib::Request& req, httplib::Response& res) { handle_generate(false, req, res); });
  svr.Post("/v1/chat/completions",
           [&](const httplib::Request& req, httplib::Response& res) { handle_generate(true, req, res); });

  PRESTO_LOG_INFO("serve", "listening on http://" + host + ":" + std::to_string(port) +
                               " (backend=" + backend->name() + ")");
  if (!svr.listen(host, port)) {
    PRESTO_LOG_ERROR("serve", "failed to bind " + host + ":" + std::to_string(port));
    return 5;
  }
  return 0;
}

} // namespace presto
