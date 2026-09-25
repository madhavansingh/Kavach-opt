#include "benchmark/benchmark_manifest.hpp"

#include <fstream>
#include <sstream>

#include "benchmark/json.hpp"

namespace kavach::benchmark {

BenchmarkSuite BenchmarkManifestLoader::load_from_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw ManifestException("Could not open manifest file: " + path.string());
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string base_dir = path.parent_path().string();
    if (base_dir.empty()) {
        base_dir = ".";
    }
    return load_from_string(buffer.str(), base_dir);
}

BenchmarkSuite BenchmarkManifestLoader::load_from_string(std::string_view content,
                                                         const std::string& base_dir) {
    JsonValue root;
    try {
        root = JsonValue::parse(content);
    } catch (const std::exception& e) {
        throw ManifestException(std::string("Malformed JSON in manifest: ") + e.what());
    }

    if (!root.is_object()) {
        throw ManifestException("Manifest root must be a JSON object");
    }

    BenchmarkSuite suite;
    if (!root.contains("suite") || !root["suite"].is_string()) {
        throw ManifestException("Manifest missing required string property 'suite'");
    }
    suite.name = root["suite"].as_string();
    suite.description = root["description"].as_string();
    suite.version = root["version"].as_string("1.0.0");

    if (!root.contains("cases") || !root["cases"].is_array()) {
        throw ManifestException("Manifest missing required array property 'cases'");
    }

    const auto& cases_arr = root["cases"].as_array();
    for (std::size_t i = 0; i < cases_arr.size(); ++i) {
        const auto& c_val = cases_arr[i];
        if (!c_val.is_object()) {
            throw ManifestException("Case item at index " + std::to_string(i) +
                                    " must be an object");
        }

        BenchmarkCase c;
        if (!c_val.contains("case_id") || !c_val["case_id"].is_string()) {
            throw ManifestException("Case at index " + std::to_string(i) +
                                    " missing string 'case_id'");
        }
        c.case_id = c_val["case_id"].as_string();

        if (c_val.contains("instance_name") && c_val["instance_name"].is_string()) {
            c.instance_name = c_val["instance_name"].as_string();
        } else {
            c.instance_name = c.case_id;
        }

        if (!c_val.contains("instance_path") || !c_val["instance_path"].is_string()) {
            throw ManifestException("Case " + c.case_id + " missing string 'instance_path'");
        }
        std::string ipath = c_val["instance_path"].as_string();
        std::filesystem::path p(ipath);
        if (p.is_relative() && !base_dir.empty() && base_dir != ".") {
            // Check if file exists relative to base_dir or repo root
            if (std::filesystem::exists(std::filesystem::path(base_dir) / p)) {
                c.instance_path = (std::filesystem::path(base_dir) / p).string();
            } else {
                c.instance_path = ipath;
            }
        } else {
            c.instance_path = ipath;
        }

        if (c_val.contains("problem_type") && c_val["problem_type"].is_string()) {
            c.problem_type = parse_problem_type(c_val["problem_type"].as_string());
        }

        if (c_val.contains("source") && c_val["source"].is_string()) {
            c.source = c_val["source"].as_string();
        }

        if (c_val.contains("tags") && c_val["tags"].is_array()) {
            for (const auto& tag_val : c_val["tags"].as_array()) {
                if (tag_val.is_string()) {
                    c.tags.push_back(tag_val.as_string());
                }
            }
        }

        if (c_val.contains("timeout_seconds") && c_val["timeout_seconds"].is_number()) {
            c.timeout_seconds = c_val["timeout_seconds"].as_double();
        }

        if (c_val.contains("reference_objective") && c_val["reference_objective"].is_number()) {
            c.reference_objective = c_val["reference_objective"].as_double();
        }

        if (c_val.contains("reference_status") && c_val["reference_status"].is_string()) {
            c.reference_status = c_val["reference_status"].as_string();
        }

        if (c_val.contains("reference_best_bound") && c_val["reference_best_bound"].is_number()) {
            c.reference_best_bound = c_val["reference_best_bound"].as_double();
        }

        if (c_val.contains("seed") && c_val["seed"].is_number()) {
            c.seed = static_cast<uint64_t>(c_val["seed"].as_int64());
        }

        if (c_val.contains("enable_presolve") && c_val["enable_presolve"].is_bool()) {
            c.enable_presolve = c_val["enable_presolve"].as_bool();
        }

        if (c_val.contains("enable_scaling") && c_val["enable_scaling"].is_bool()) {
            c.enable_scaling = c_val["enable_scaling"].as_bool();
        }

        if (c_val.contains("spmv_batch_size") && c_val["spmv_batch_size"].is_number()) {
            c.spmv_batch_size = static_cast<std::size_t>(c_val["spmv_batch_size"].as_int64());
        }

        suite.cases.push_back(std::move(c));
    }

    return suite;
}

}  // namespace kavach::benchmark
