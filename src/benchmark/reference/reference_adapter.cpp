#include "benchmark/reference/reference_adapter.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace kavach::benchmark {

std::string to_string(ReferenceSolverStatus s) {
    switch (s) {
        case ReferenceSolverStatus::OPTIMAL:
            return "OPTIMAL";
        case ReferenceSolverStatus::FEASIBLE:
            return "FEASIBLE";
        case ReferenceSolverStatus::INFEASIBLE:
            return "INFEASIBLE";
        case ReferenceSolverStatus::UNBOUNDED:
            return "UNBOUNDED";
        case ReferenceSolverStatus::TIMEOUT:
            return "TIMEOUT";
        case ReferenceSolverStatus::UNAVAILABLE:
            return "UNAVAILABLE";
        case ReferenceSolverStatus::ERROR:
            return "ERROR";
    }
    return "ERROR";
}

ExternalReferenceAdapter::ExternalReferenceAdapter(std::string custom_bin_path)
    : binary_path_(std::move(custom_bin_path)) {
    if (binary_path_.empty()) {
        binary_path_ = discover_binary();
    }
}

bool ExternalReferenceAdapter::is_available() const noexcept {
    if (binary_path_.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(binary_path_, ec);
}

std::string ExternalReferenceAdapter::discover_binary() {
    // 1. Check environment variables
    const char* env1 = std::getenv("HIGHS_BIN");
    if (env1 && *env1) {
        std::error_code ec;
        if (std::filesystem::exists(env1, ec)) {
            return env1;
        }
    }
    const char* env2 = std::getenv("HIGHS_PATH");
    if (env2 && *env2) {
        std::error_code ec;
        if (std::filesystem::exists(env2, ec)) {
            return env2;
        }
    }

    // 2. Check standard system and user-level bin locations
    std::vector<std::string> standard_paths = {"/opt/homebrew/bin/highs", "/usr/local/bin/highs",
                                               "/usr/bin/highs", "/bin/highs",
                                               "C:\\Program Files\\HiGHS\\bin\\highs.exe"};

    if (const char* home = std::getenv("HOME"); home && *home) {
        const std::string h(home);
        standard_paths.push_back(h + "/.local/bin/highs");
        standard_paths.push_back(h + "/Library/Python/3.9/bin/highs");
        standard_paths.push_back(h + "/Library/Python/3.10/bin/highs");
        standard_paths.push_back(h + "/Library/Python/3.11/bin/highs");
        standard_paths.push_back(h + "/Library/Python/3.12/bin/highs");
    }

    for (const auto& p : standard_paths) {
        std::error_code ec;
        if (std::filesystem::exists(p, ec)) {
            return p;
        }
    }

#if defined(__unix__) || defined(__APPLE__)
    // 3. Fallback check with which
    FILE* pipe = popen("which highs 2>/dev/null", "r");
    if (pipe) {
        char buffer[256];
        std::string result;
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result = buffer;
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }
        }
        pclose(pipe);
        if (!result.empty()) {
            std::error_code ec;
            if (std::filesystem::exists(result, ec)) {
                return result;
            }
        }
    }
#endif

    return "";
}

ReferenceRunResult ExternalReferenceAdapter::parse_output(std::string_view stdout_str,
                                                          std::string_view stderr_str,
                                                          int exit_code, bool timed_out) {
    ReferenceRunResult res;
    res.raw_output = std::string(stdout_str) + "\n" + std::string(stderr_str);

    if (timed_out) {
        res.status = ReferenceSolverStatus::TIMEOUT;
        res.error_message = "HiGHS execution exceeded time limit.";
        return res;
    }

    std::string out_copy(stdout_str);
    std::istringstream iss(out_copy);
    std::string line;
    bool found_status = false;

    while (std::getline(iss, line)) {
        // Version
        if (line.find("Running HiGHS") != std::string::npos) {
            auto pos = line.find("Running HiGHS");
            res.solver_version = line.substr(pos);
        }

        // Status
        if (line.find("Model   status      :") != std::string::npos ||
            line.find("Model status        :") != std::string::npos) {
            if (line.find("Optimal") != std::string::npos) {
                res.status = ReferenceSolverStatus::OPTIMAL;
                found_status = true;
            } else if (line.find("Infeasible") != std::string::npos) {
                res.status = ReferenceSolverStatus::INFEASIBLE;
                found_status = true;
            } else if (line.find("Unbounded") != std::string::npos) {
                res.status = ReferenceSolverStatus::UNBOUNDED;
                found_status = true;
            } else if (line.find("Time limit") != std::string::npos) {
                res.status = ReferenceSolverStatus::TIMEOUT;
                found_status = true;
            } else if (line.find("Feasible") != std::string::npos) {
                res.status = ReferenceSolverStatus::FEASIBLE;
                found_status = true;
            }
        }

        // Objective value
        if (line.find("Objective value     :") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                try {
                    res.objective = std::stod(line.substr(pos + 1));
                } catch (...) {
                }
            }
        }

        // Runtime
        if (line.find("HiGHS run time      :") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                try {
                    res.runtime_seconds = std::stod(line.substr(pos + 1));
                } catch (...) {
                }
            }
        }

        // Simplex iterations
        if (line.find("Simplex   iterations:") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                try {
                    res.iterations = std::stoull(line.substr(pos + 1));
                } catch (...) {
                }
            }
        }

        // MIP nodes
        if (line.find("MIP nodes           :") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                try {
                    res.nodes = std::stoull(line.substr(pos + 1));
                } catch (...) {
                }
            }
        }

        // MIP gap
        if (line.find("MIP gap             :") != std::string::npos) {
            auto pos = line.find(':');
            if (pos != std::string::npos && pos + 1 < line.size()) {
                try {
                    res.gap = std::stod(line.substr(pos + 1)) / 100.0;
                } catch (...) {
                }
            }
        }
    }

    if (!found_status) {
        if (exit_code == 0 && res.objective.has_value()) {
            res.status = ReferenceSolverStatus::OPTIMAL;
        } else if (exit_code != 0) {
            res.status = ReferenceSolverStatus::ERROR;
            res.error_message =
                "Process exited with code " + std::to_string(exit_code) + ": " + res.raw_output;
        } else {
            res.status = ReferenceSolverStatus::ERROR;
            res.error_message = "Unrecognized solver output format.";
        }
    }

    return res;
}

ReferenceRunResult ExternalReferenceAdapter::run(const std::string& model_path,
                                                 double time_limit_seconds) const {
    if (!is_available()) {
        ReferenceRunResult r;
        r.status = ReferenceSolverStatus::UNAVAILABLE;
        r.error_message = "HiGHS binary not found in PATH or HIGHS_BIN.";
        return r;
    }

#if defined(__unix__) || defined(__APPLE__)
    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        ReferenceRunResult r;
        r.status = ReferenceSolverStatus::ERROR;
        r.error_message = "Failed to create pipes for subprocess.";
        return r;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        ReferenceRunResult r;
        r.status = ReferenceSolverStatus::ERROR;
        r.error_message = "Failed to fork subprocess.";
        return r;
    }

    if (pid == 0) {
        // Child process
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::string time_arg = "--time_limit=" + std::to_string(time_limit_seconds);
        const char* args[] = {binary_path_.c_str(), model_path.c_str(), time_arg.c_str(), nullptr};
        execv(binary_path_.c_str(), const_cast<char* const*>(args));
        _exit(127);
    }

    // Parent process
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Set non-blocking on read ends
    fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

    std::string stdout_str;
    std::string stderr_str;
    char buffer[4096];

    auto start_time = std::chrono::steady_clock::now();
    bool timed_out = false;
    int status = 0;
    bool child_exited = false;

    while (true) {
        // Read available stdout
        ssize_t bytes = read(stdout_pipe[0], buffer, sizeof(buffer));
        if (bytes > 0) {
            stdout_str.append(buffer, static_cast<std::size_t>(bytes));
        }

        // Read available stderr
        bytes = read(stderr_pipe[0], buffer, sizeof(buffer));
        if (bytes > 0) {
            stderr_str.append(buffer, static_cast<std::size_t>(bytes));
        }

        pid_t res = waitpid(pid, &status, WNOHANG);
        if (res == pid) {
            child_exited = true;
            // Drain remaining pipe
            while ((bytes = read(stdout_pipe[0], buffer, sizeof(buffer))) > 0) {
                stdout_str.append(buffer, static_cast<std::size_t>(bytes));
            }
            while ((bytes = read(stderr_pipe[0], buffer, sizeof(buffer))) > 0) {
                stderr_str.append(buffer, static_cast<std::size_t>(bytes));
            }
            break;
        }

        auto elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        if (time_limit_seconds > 0.0 && elapsed > time_limit_seconds) {
            timed_out = true;
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    int exit_code = child_exited && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return parse_output(stdout_str, stderr_str, exit_code, timed_out);
#else
    ReferenceRunResult r;
    r.status = ReferenceSolverStatus::UNAVAILABLE;
    r.error_message = "Subprocess execution not supported on this platform.";
    return r;
#endif
}

}  // namespace kavach::benchmark
