// presto - OpenAI-compatible HTTP server (cpp-httplib based)
#include "presto/server.hpp"

#include "json_mini.hpp"
#include "presto/backend.hpp"
#include "presto/engine.hpp"
#include "presto/log.hpp"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <string>
#include <thread>
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
  std::vector<ChatMessage> chat_messages;
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
    for (const auto& m : msgs->items()) {
      const json::Node* role = m.find("role");
      const json::Node* content = m.find("content");
      if (!role || !role->is_string() || !content || !content->is_string()) {
        err = "each message requires string role and content";
        return false;
      }
      opts.chat_messages.push_back({role->as_string(), content->as_string()});
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
  auto env_int_local = [](const char* k, int dflt) {
    if (const char* v = std::getenv(k)) { try { return std::stoi(v); } catch (...) {} }
    return dflt;
  };
  const int batch_slots = std::max(1, env_int_local("PRESTO_BATCH_SLOTS", 4));

  // one worker thread owns the engine; concurrent requests are gathered so
  // N users share each weight-streaming decode via generate_many()
  struct PendingJob {
    GenerateParams params;
    GenerateResult result;
    std::string err;
    bool ok = false;
    bool done = false;
  };
  std::mutex qm;
  std::condition_variable cv_q, cv_done;
  std::deque<std::shared_ptr<PendingJob>> queue;
  bool worker_stop = false;
  std::thread worker([&] {
    for (;;) {
      std::vector<std::shared_ptr<PendingJob>> take;
      {
        std::unique_lock<std::mutex> lk(qm);
        cv_q.wait(lk, [&] { return worker_stop || !queue.empty(); });
        if (worker_stop && queue.empty()) return;
        // coalescing window: give simultaneous arrivals time to land in
        // the queue so one engine pass can carry them together
        if (!worker_stop)
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        while (!queue.empty() && static_cast<int>(take.size()) < batch_slots) {
          take.push_back(std::move(queue.front()));
          queue.pop_front();
        }
      }
      PRESTO_LOG_INFO("serve", "gathered " + std::to_string(take.size()) +
                      " request(s) for one engine pass");
      if (take.size() > 1) {
        std::vector<BatchJob> jobs(take.size());
        for (size_t i = 0; i < take.size(); ++i) jobs[i].params = take[i]->params;
        if (!backend->generate_many(jobs, err)) {
          for (auto& j : jobs) j.ok = backend->generate(j.params, j.result, j.err);
        }
        for (size_t i = 0; i < take.size(); ++i) {
          take[i]->ok = jobs[i].ok;
          take[i]->result = std::move(jobs[i].result);
          take[i]->err = std::move(jobs[i].err);
          take[i]->done = true;
        }
      } else {
        auto& job = *take[0];
        job.ok = backend->generate(job.params, job.result, job.err);
        job.done = true;
      }
      cv_done.notify_all();
    }
  });
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
    RequestOpts opts;
    std::string perr;
    if (!parse_request_body(req.body, chat, opts, perr)) {
      res.status = 400;
      res.set_content(R"({"error":{"message":")" + json_escape(perr) + R"("}})",
                      "application/json");
      return;
    }
    auto job = std::make_shared<PendingJob>();
    job->params.prompt_text = opts.prompt_text;
    job->params.prompt_tokens = opts.prompt_tokens;
    job->params.chat_messages = opts.chat_messages;
    job->params.max_tokens = opts.max_tokens;
    job->params.temp = opts.temp;
    {
      std::lock_guard<std::mutex> lk(qm);
      queue.push_back(job);
    }
    cv_q.notify_one();
    {
      std::unique_lock<std::mutex> lk(qm);
      cv_done.wait(lk, [&] { return job->done; });
    }

    GenerateResult& r = job->result;
    if (!job->ok) {
      PRESTO_LOG_ERROR("serve", "generate failed: " + job->err);
      res.status = 500;
      res.set_content(R"({"error":{"message":")" + json_escape(job->err) +
                          R"("}})",
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
