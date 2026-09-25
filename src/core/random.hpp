#pragma once

#include <cstdint>
#include <random>

namespace kavach::core {

/**
 * @brief Deterministic pseudo-random generator conforming to Section 23 Reproducibility.
 *
 * Ensures byte-for-byte reproducibility across runs with the same seed.
 */
class DeterministicRNG {
public:
    explicit DeterministicRNG(uint64_t seed = 42);

    void set_seed(uint64_t seed);
    [[nodiscard]] uint64_t get_seed() const noexcept;

    [[nodiscard]] double uniform_real(double min = 0.0, double max = 1.0);
    [[nodiscard]] int uniform_int(int min, int max);

private:
    uint64_t seed_{42};
    std::mt19937_64 engine_;
};

}  // namespace kavach::core
