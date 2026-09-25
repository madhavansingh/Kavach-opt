#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "core/logger.hpp"
#include "core/random.hpp"
#include "core/version.hpp"

namespace {

void print_version() {
    std::cout << kavach::core::PROJECT_NAME << " version " << kavach::core::PROJECT_VERSION << "\n"
              << "Build Type: " << kavach::core::BUILD_TYPE << "\n"
              << "Compiler: " << kavach::core::COMPILER_ID << " " << kavach::core::COMPILER_VERSION
              << "\n"
              << "Commit: " << kavach::core::GIT_COMMIT_HASH << "\n"
              << "Sovereign Optimization Solver Core (SIH26119)\n";
}

void print_help() {
    std::cout << "Kavach-Opt — Indigenous GPU-Accelerated Optimization Solver Core\n\n"
              << "Usage:\n"
              << "  kavach-opt solve <model.mps|.lp> [options]\n"
              << "  kavach-opt benchmark --suite <netlib|miplib-subset> [options]\n"
              << "  kavach-opt verify <solution.json>\n"
              << "  kavach-opt presolve <model.mps> --report\n"
              << "  kavach-opt --version\n"
              << "  kavach-opt --help\n\n"
              << "Options:\n"
              << "  --problem-type <lp|milp|qp>   Explicitly specify problem type\n"
              << "  --gpu                         Enable scoped GPU batched-SpMV kernel\n"
              << "  --no-gpu                      Force CPU execution (default)\n"
              << "  --time-limit <seconds>        Wall-clock time limit\n"
              << "  --seed <uint64>               Random seed for deterministic reproducibility\n"
              << "  --verbose                     Enable debug output\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string_view> args(argv + 1, argv + argc);

        if (args.empty() || args[0] == "--help" || args[0] == "-h") {
            print_help();
            return 0;
        }

        if (args[0] == "--version" || args[0] == "-v") {
            print_version();
            return 0;
        }

        std::string_view command = args[0];
        if (command == "solve" || command == "benchmark" || command == "verify" ||
            command == "presolve") {
            KAVACH_LOG_INFO(
                "Kavach-Opt Phase 0 Engineering Foundation active. "
                "Solver algorithms are scheduled in gated Phases 1-15 per Master Blueprint v2.");
            std::cout << "Status: Phase 0 Foundation initialized. Use --version or --help.\n";
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\nRun 'kavach-opt --help' for usage.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
        return 1;
    }
}
