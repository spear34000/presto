// presto - leveled logger (header-only)
// Env override: PRESTO_LOG=debug|info|warn|error (default: info)
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>

namespace presto::log {

enum class Level : int { Debug = 0, Info = 1, Warn = 2, Error = 3 };

inline const char* level_name(Level l) {
  switch (l) {
    case Level::Debug: return "DEBUG";
    case Level::Info: return "INFO";
    case Level::Warn: return "WARN";
    case Level::Error: return "ERROR";
  }
  return "?";
}

inline Level min_level() {
  static const Level lvl = [] {
    const char* e = std::getenv("PRESTO_LOG");
    if (!e || !*e) return Level::Info;
    std::string s(e);
    for (auto& c : s) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    if (s == "debug") return Level::Debug;
    if (s == "warn") return Level::Warn;
    if (s == "error") return Level::Error;
    return Level::Info;
  }();
  return lvl;
}

inline void emit(Level lvl, const char* module, const std::string& msg) {
  if (static_cast<int>(lvl) < static_cast<int>(min_level())) return;
  const auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char ts[16]{};
  std::strftime(ts, sizeof ts, "%H:%M:%S", &tm);
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
  std::ostringstream o;
  o << "[presto][" << level_name(lvl) << "][" << module << "] " << ts << '.' << ms << " | " << msg;
  std::fprintf(stderr, "%s\n", o.str().c_str());
}

} // namespace presto::log

#define PRESTO_LOG(lvl, mod, msg) ::presto::log::emit(::presto::log::Level::lvl, mod, (msg))
#define PRESTO_LOG_DEBUG(mod, msg) PRESTO_LOG(Debug, mod, msg)
#define PRESTO_LOG_INFO(mod, msg) PRESTO_LOG(Info, mod, msg)
#define PRESTO_LOG_WARN(mod, msg) PRESTO_LOG(Warn, mod, msg)
#define PRESTO_LOG_ERROR(mod, msg) PRESTO_LOG(Error, mod, msg)
