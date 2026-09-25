#include "core/random.hpp"

namespace kavach::core {

DeterministicRNG::DeterministicRNG(uint64_t seed) : seed_(seed), engine_(seed) {}

void DeterministicRNG::set_seed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

uint64_t DeterministicRNG::get_seed() const noexcept {
    return seed_;
}

double DeterministicRNG::uniform_real(double min, double max) {
    std::uniform_real_distribution<double> dist(min, max);
    return dist(engine_);
}

int DeterministicRNG::uniform_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(engine_);
}

}  // namespace kavach::core
