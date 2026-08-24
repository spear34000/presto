// presto - model path resolution
#include "presto/resolve.hpp"

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace presto {
namespace {

std::vector<fs::path> library_roots() {
    std::vector<fs::path> roots;
    roots.push_back(fs::current_path() / "models");
    if (const char* lad = std::getenv("LOCALAPPDATA")) {
        roots.push_back(fs::path(lad) / "presto" / "models");
    }
    if (const char* home = std::getenv("USERPROFILE")) {
        roots.push_back(fs::path(home) / ".lmstudio" / "models");
    } else if (const char* h = std::getenv("HOME")) {
        roots.push_back(fs::path(h) / ".lmstudio" / "models");
        roots.push_back(fs::path(h) / ".presto" / "models");
    }
    return roots;
}

bool is_model_file(const fs::path& p) {
    std::string ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(::tolower(c));
    return ext == ".gguf" || ext == ".safetensors";
}

} // namespace

std::string resolve_model_path(const std::string& ref) {
    std::error_code ec;
    if (fs::exists(fs::path(ref), ec)) return ref;

    const fs::path refp(ref);
    const std::string lower_ref = [&] {
        std::string s = refp.filename().string();
        for (auto& c : s) c = static_cast<char>(::tolower(c));
        return s;
    }();

    // recursive pass over library roots: direct root/<ref> hit first, then
    // exact filename match anywhere, then unique prefix match
    std::string prefix_hit;
    for (const fs::path& root : library_roots()) {
        if (!fs::exists(root, ec)) continue;
        const fs::path direct = root / ref;
        if (fs::exists(direct, ec)) return direct.string();
        for (const auto& entry : fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file(ec) || !is_model_file(entry.path())) continue;
            const std::string fname = entry.path().filename().string();
            std::string fl = fname;
            for (auto& c : fl) c = static_cast<char>(::tolower(c));
            if (fl == lower_ref) return entry.path().string();
            if (prefix_hit.empty() && fl.rfind(lower_ref, 0) == 0)
                prefix_hit = entry.path().string();
        }
    }
    return prefix_hit;
}

std::vector<std::string> list_known_models() {
    std::vector<std::string> out;
    std::error_code ec;
    for (const fs::path& root : library_roots()) {
        if (!fs::exists(root, ec)) continue;
        for (const auto& entry : fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec)) {
            if (entry.is_regular_file(ec) && is_model_file(entry.path()))
                out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace presto
