#include "benchmark/benchmark_io.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace kavach::benchmark {

namespace {

template <typename T>
JsonValue opt_to_json(const std::optional<T>& opt) {
    if (opt.has_value()) {
        return JsonValue(*opt);
    }
    return JsonValue(nullptr);
}

JsonValue opt_double_to_json(const std::optional<double>& opt) {
    if (opt.has_value()) {
        return JsonValue(*opt);
    }
    return JsonValue(nullptr);
}

JsonValue opt_size_to_json(const std::optional<std::size_t>& opt) {
    if (opt.has_value()) {
        return JsonValue(static_cast<int64_t>(*opt));
    }
    return JsonValue(nullptr);
}

}  // namespace

JsonValue BenchmarkIO::to_json(const ReproducibilityManifest& m) {
    JsonValue::Object obj;
    obj["git_commit"] = m.git_commit;
    obj["timestamp"] = m.timestamp;
    obj["hostname"] = m.hostname;
    obj["cpu_model"] = m.cpu_model;
    obj["num_cpu_cores"] = static_cast<int64_t>(m.num_cpu_cores);
    obj["gpu_device"] = m.gpu_device;
    obj["os_name"] = m.os_name;
    obj["compiler_info"] = m.compiler_info;
    obj["build_type"] = m.build_type;
    obj["kavach_version"] = m.kavach_version;
    obj["suite_name"] = m.suite_name;
    obj["manifest_version"] = m.manifest_version;
    obj["seed"] = static_cast<int64_t>(m.seed);
    obj["reference_solver_info"] = m.reference_solver_info;
    return JsonValue(std::move(obj));
}

JsonValue BenchmarkIO::to_json(const BenchmarkResult& r) {
    JsonValue::Object obj;
    obj["timestamp"] = r.timestamp;
    obj["suite"] = r.suite;
    obj["case_id"] = r.case_id;
    obj["instance_name"] = r.instance_name;
    obj["instance_path"] = r.instance_path;
    obj["problem_type"] = r.problem_type;

    JsonValue::Array tags_arr;
    for (const auto& t : r.tags) {
        tags_arr.push_back(t);
    }
    obj["tags"] = JsonValue(std::move(tags_arr));

    obj["variables"] = opt_size_to_json(r.variables);
    obj["constraints"] = opt_size_to_json(r.constraints);
    obj["nonzeros"] = opt_size_to_json(r.nonzeros);
    obj["density"] = opt_double_to_json(r.density);

    obj["solver_name"] = r.solver_name;
    obj["solver_version"] = r.solver_version;
    obj["status"] = to_string(r.status);
    obj["verified_status"] = r.verified_status;
    obj["verification_passed"] = r.verification_passed;

    obj["objective_value"] = opt_double_to_json(r.objective_value);
    obj["reference_objective"] = opt_double_to_json(r.reference_objective);
    obj["reference_status"] = opt_to_json(r.reference_status);
    obj["objective_error_abs"] = opt_double_to_json(r.objective_error_abs);
    obj["objective_error_rel"] = opt_double_to_json(r.objective_error_rel);
    obj["best_bound"] = opt_double_to_json(r.best_bound);
    obj["optimality_gap"] = opt_double_to_json(r.optimality_gap);

    obj["iterations"] = opt_size_to_json(r.iterations);
    obj["nodes"] = opt_size_to_json(r.nodes);

    obj["total_wall_time_seconds"] = r.total_wall_time_seconds;
    obj["parse_time_seconds"] = opt_double_to_json(r.parse_time_seconds);
    obj["presolve_time_seconds"] = opt_double_to_json(r.presolve_time_seconds);
    obj["solve_time_seconds"] = opt_double_to_json(r.solve_time_seconds);
    obj["verification_time_seconds"] = opt_double_to_json(r.verification_time_seconds);

    obj["primal_residual"] = opt_double_to_json(r.primal_residual);
    obj["dual_residual"] = opt_double_to_json(r.dual_residual);
    obj["kkt_residual"] = opt_double_to_json(r.kkt_residual);
    obj["integrality_violation"] = opt_double_to_json(r.integrality_violation);

    if (r.memory_bytes.has_value()) {
        obj["memory_bytes"] = JsonValue(*r.memory_bytes);
    } else {
        obj["memory_bytes"] = JsonValue(nullptr);
    }

    obj["backend"] = r.backend;
    obj["num_threads"] = static_cast<int64_t>(r.num_threads);
    obj["gpu_device"] = opt_to_json(r.gpu_device);

    if (r.gpu_telemetry.has_value()) {
        JsonValue::Object g;
        g["gpu_available"] = r.gpu_telemetry->gpu_available;
        g["fallback_occurred"] = r.gpu_telemetry->fallback_occurred;
        g["fallback_reason"] = r.gpu_telemetry->fallback_reason;
        g["device_name"] = r.gpu_telemetry->device_name;
        g["h2d_time_seconds"] = r.gpu_telemetry->h2d_time_seconds;
        g["kernel_time_seconds"] = r.gpu_telemetry->kernel_time_seconds;
        g["d2h_time_seconds"] = r.gpu_telemetry->d2h_time_seconds;
        g["total_gpu_time_seconds"] = r.gpu_telemetry->total_gpu_time_seconds;
        g["cpu_reference_time_seconds"] = r.gpu_telemetry->cpu_reference_time_seconds;
        g["transfer_inclusive_speedup"] = r.gpu_telemetry->transfer_inclusive_speedup;
        g["kernel_only_speedup"] = r.gpu_telemetry->kernel_only_speedup;
        obj["gpu_telemetry"] = JsonValue(std::move(g));
    } else {
        obj["gpu_telemetry"] = JsonValue(nullptr);
    }

    obj["cpu_details"] = r.cpu_details;
    obj["compiler"] = r.compiler;
    obj["build_type"] = r.build_type;
    obj["git_commit"] = r.git_commit;
    obj["random_seed"] = static_cast<int64_t>(r.random_seed);
    obj["error_message"] = r.error_message;

    if (r.has_reference_comparison) {
        JsonValue::Object ref;
        ref["solver_name"] = r.reference_solver_name;
        ref["solver_version"] = opt_to_json(r.reference_solver_version);
        ref["status"] = opt_to_json(r.reference_solver_status);
        ref["objective"] = opt_double_to_json(r.reference_objective_val);
        ref["runtime_seconds"] = opt_double_to_json(r.reference_runtime_seconds);
        ref["iterations"] = opt_size_to_json(r.reference_iterations);
        ref["gap"] = opt_double_to_json(r.reference_gap);
        ref["error_message"] = r.reference_error_message;
        obj["reference_comparison"] = JsonValue(std::move(ref));
    } else {
        obj["reference_comparison"] = JsonValue(nullptr);
    }

    return JsonValue(std::move(obj));
}

BenchmarkResult BenchmarkIO::from_json(const JsonValue& j) {
    BenchmarkResult r;
    r.timestamp = j["timestamp"].as_string();
    r.suite = j["suite"].as_string();
    r.case_id = j["case_id"].as_string();
    r.instance_name = j["instance_name"].as_string();
    r.instance_path = j["instance_path"].as_string();
    r.problem_type = j["problem_type"].as_string();

    if (j.contains("tags") && j["tags"].is_array()) {
        for (const auto& t : j["tags"].as_array()) {
            r.tags.push_back(t.as_string());
        }
    }

    if (!j["variables"].is_null()) {
        r.variables = static_cast<std::size_t>(j["variables"].as_int64());
    }
    if (!j["constraints"].is_null()) {
        r.constraints = static_cast<std::size_t>(j["constraints"].as_int64());
    }
    if (!j["nonzeros"].is_null()) {
        r.nonzeros = static_cast<std::size_t>(j["nonzeros"].as_int64());
    }
    if (!j["density"].is_null()) {
        r.density = j["density"].as_double();
    }

    r.solver_name = j["solver_name"].as_string();
    r.solver_version = j["solver_version"].as_string();
    r.status = parse_benchmark_status(j["status"].as_string());
    r.verified_status = j["verified_status"].as_string();
    r.verification_passed = j["verification_passed"].as_bool();

    if (!j["objective_value"].is_null()) {
        r.objective_value = j["objective_value"].as_double();
    }
    if (!j["reference_objective"].is_null()) {
        r.reference_objective = j["reference_objective"].as_double();
    }
    if (!j["reference_status"].is_null()) {
        r.reference_status = j["reference_status"].as_string();
    }
    if (!j["objective_error_abs"].is_null()) {
        r.objective_error_abs = j["objective_error_abs"].as_double();
    }
    if (!j["objective_error_rel"].is_null()) {
        r.objective_error_rel = j["objective_error_rel"].as_double();
    }
    if (!j["best_bound"].is_null()) {
        r.best_bound = j["best_bound"].as_double();
    }
    if (!j["optimality_gap"].is_null()) {
        r.optimality_gap = j["optimality_gap"].as_double();
    }

    if (!j["iterations"].is_null()) {
        r.iterations = static_cast<std::size_t>(j["iterations"].as_int64());
    }
    if (!j["nodes"].is_null()) {
        r.nodes = static_cast<std::size_t>(j["nodes"].as_int64());
    }

    r.total_wall_time_seconds = j["total_wall_time_seconds"].as_double();
    if (!j["parse_time_seconds"].is_null()) {
        r.parse_time_seconds = j["parse_time_seconds"].as_double();
    }
    if (!j["presolve_time_seconds"].is_null()) {
        r.presolve_time_seconds = j["presolve_time_seconds"].as_double();
    }
    if (!j["solve_time_seconds"].is_null()) {
        r.solve_time_seconds = j["solve_time_seconds"].as_double();
    }
    if (!j["verification_time_seconds"].is_null()) {
        r.verification_time_seconds = j["verification_time_seconds"].as_double();
    }

    if (!j["primal_residual"].is_null()) {
        r.primal_residual = j["primal_residual"].as_double();
    }
    if (!j["dual_residual"].is_null()) {
        r.dual_residual = j["dual_residual"].as_double();
    }
    if (!j["kkt_residual"].is_null()) {
        r.kkt_residual = j["kkt_residual"].as_double();
    }
    if (!j["integrality_violation"].is_null()) {
        r.integrality_violation = j["integrality_violation"].as_double();
    }

    if (!j["memory_bytes"].is_null()) {
        r.memory_bytes = j["memory_bytes"].as_int64();
    }

    r.backend = j["backend"].as_string();
    r.num_threads = static_cast<std::size_t>(j["num_threads"].as_int64(1));
    if (!j["gpu_device"].is_null()) {
        r.gpu_device = j["gpu_device"].as_string();
    }

    if (j.contains("gpu_telemetry") && !j["gpu_telemetry"].is_null()) {
        const auto& g = j["gpu_telemetry"];
        backend::cuda::CudaTelemetry tel;
        tel.gpu_available = g["gpu_available"].as_bool();
        tel.fallback_occurred = g["fallback_occurred"].as_bool();
        tel.fallback_reason = g["fallback_reason"].as_string();
        tel.device_name = g["device_name"].as_string();
        tel.h2d_time_seconds = g["h2d_time_seconds"].as_double();
        tel.kernel_time_seconds = g["kernel_time_seconds"].as_double();
        tel.d2h_time_seconds = g["d2h_time_seconds"].as_double();
        tel.total_gpu_time_seconds = g["total_gpu_time_seconds"].as_double();
        tel.cpu_reference_time_seconds = g["cpu_reference_time_seconds"].as_double();
        tel.transfer_inclusive_speedup = g["transfer_inclusive_speedup"].as_double();
        tel.kernel_only_speedup = g["kernel_only_speedup"].as_double();
        r.gpu_telemetry = tel;
    }

    r.cpu_details = j["cpu_details"].as_string();
    r.compiler = j["compiler"].as_string();
    r.build_type = j["build_type"].as_string();
    r.git_commit = j["git_commit"].as_string();
    r.random_seed = static_cast<uint64_t>(j["random_seed"].as_int64(42));
    r.error_message = j["error_message"].as_string();

    if (j.contains("reference_comparison") && !j["reference_comparison"].is_null()) {
        r.has_reference_comparison = true;
        const auto& ref = j["reference_comparison"];
        r.reference_solver_name = ref["solver_name"].as_string();
        if (!ref["solver_version"].is_null()) {
            r.reference_solver_version = ref["solver_version"].as_string();
        }
        if (!ref["status"].is_null()) {
            r.reference_solver_status = ref["status"].as_string();
        }
        if (!ref["objective"].is_null()) {
            r.reference_objective_val = ref["objective"].as_double();
        }
        if (!ref["runtime_seconds"].is_null()) {
            r.reference_runtime_seconds = ref["runtime_seconds"].as_double();
        }
        if (!ref["iterations"].is_null()) {
            r.reference_iterations = static_cast<std::size_t>(ref["iterations"].as_int64());
        }
        if (!ref["gap"].is_null()) {
            r.reference_gap = ref["gap"].as_double();
        }
        r.reference_error_message = ref["error_message"].as_string();
    }

    return r;
}

void BenchmarkIO::export_json(const std::vector<BenchmarkResult>& results,
                              const ReproducibilityManifest& manifest,
                              const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    JsonValue::Object root;
    root["manifest"] = to_json(manifest);

    JsonValue::Array res_arr;
    for (const auto& r : results) {
        res_arr.push_back(to_json(r));
    }
    root["results"] = JsonValue(std::move(res_arr));

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    ofs << JsonValue(std::move(root)).dump(2) << '\n';
}

std::pair<ReproducibilityManifest, std::vector<BenchmarkResult>> BenchmarkIO::import_json(
    const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path.string());
    }
    std::stringstream ss;
    ss << ifs.rdbuf();

    JsonValue root = JsonValue::parse(ss.str());

    ReproducibilityManifest manifest;
    if (root.contains("manifest")) {
        const auto& m = root["manifest"];
        manifest.git_commit = m["git_commit"].as_string();
        manifest.timestamp = m["timestamp"].as_string();
        manifest.hostname = m["hostname"].as_string();
        manifest.cpu_model = m["cpu_model"].as_string();
        manifest.num_cpu_cores = static_cast<std::size_t>(m["num_cpu_cores"].as_int64(1));
        manifest.gpu_device = m["gpu_device"].as_string();
        manifest.os_name = m["os_name"].as_string();
        manifest.compiler_info = m["compiler_info"].as_string();
        manifest.build_type = m["build_type"].as_string();
        manifest.kavach_version = m["kavach_version"].as_string();
        manifest.suite_name = m["suite_name"].as_string();
        manifest.manifest_version = m["manifest_version"].as_string();
        manifest.seed = static_cast<uint64_t>(m["seed"].as_int64(42));
        manifest.reference_solver_info = m["reference_solver_info"].as_string();
    }

    std::vector<BenchmarkResult> results;
    if (root.contains("results") && root["results"].is_array()) {
        for (const auto& val : root["results"].as_array()) {
            results.push_back(from_json(val));
        }
    }

    return {manifest, results};
}

void BenchmarkIO::export_csv(const std::vector<BenchmarkResult>& results,
                             const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open CSV for writing: " + path.string());
    }

    // Header
    ofs << "case_id,instance,suite,problem_type,status,verified_status,objective,"
           "reference_objective,rel_error,total_time_s,solve_time_s,iterations,nodes,"
           "primal_residual,dual_residual,kkt_residual,ref_status,ref_objective,ref_time_s\n";

    for (const auto& r : results) {
        ofs << r.case_id << ',' << r.instance_name << ',' << r.suite << ',' << r.problem_type << ','
            << to_string(r.status) << ',' << r.verified_status << ',';

        if (r.objective_value.has_value()) {
            ofs << std::setprecision(12) << *r.objective_value;
        }
        ofs << ',';

        if (r.reference_objective.has_value()) {
            ofs << std::setprecision(12) << *r.reference_objective;
        }
        ofs << ',';

        if (r.objective_error_rel.has_value()) {
            ofs << std::setprecision(6) << *r.objective_error_rel;
        }
        ofs << ',';

        ofs << std::fixed << std::setprecision(6) << r.total_wall_time_seconds << ',';

        if (r.solve_time_seconds.has_value()) {
            ofs << std::fixed << std::setprecision(6) << *r.solve_time_seconds;
        }
        ofs << ',';

        if (r.iterations.has_value()) {
            ofs << *r.iterations;
        }
        ofs << ',';

        if (r.nodes.has_value()) {
            ofs << *r.nodes;
        }
        ofs << ',';

        if (r.primal_residual.has_value()) {
            ofs << std::scientific << std::setprecision(4) << *r.primal_residual
                << std::defaultfloat;
        }
        ofs << ',';

        if (r.dual_residual.has_value()) {
            ofs << std::scientific << std::setprecision(4) << *r.dual_residual << std::defaultfloat;
        }
        ofs << ',';

        if (r.kkt_residual.has_value()) {
            ofs << std::scientific << std::setprecision(4) << *r.kkt_residual << std::defaultfloat;
        }
        ofs << ',';

        if (r.has_reference_comparison) {
            if (r.reference_solver_status.has_value()) {
                ofs << *r.reference_solver_status;
            }
            ofs << ',';
            if (r.reference_objective_val.has_value()) {
                ofs << std::setprecision(12) << *r.reference_objective_val;
            }
            ofs << ',';
            if (r.reference_runtime_seconds.has_value()) {
                ofs << std::fixed << std::setprecision(6) << *r.reference_runtime_seconds;
            }
        } else {
            ofs << ",,";
        }
        ofs << '\n';
    }
}

}  // namespace kavach::benchmark
