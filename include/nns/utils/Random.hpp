#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <any>
#include <ranges>

#include <Eigen/Dense>

#include <nns/core/Types.hpp>
namespace nns {
enum class Distribution { Normal, XavierNormal, HeNormal };

struct GainTag{};
using Gain = StrongType<GainTag>;
enum Seed : uint32_t;

class RandomGenerator {
public:
    RandomGenerator(Seed seed = Seed{738547485u}) : gen_(seed), seed_(seed) {
    }

    void reseed(uint32_t seed) {
        seed_ = seed;
        gen_.seed(seed);
    }

    uint32_t seed() const noexcept {
        return seed_;
    }

    void init_matrix(Matrix& mat, Distribution dist = Distribution::Normal, Gain gain = Gain{1.0}) {
        double std_dev;
        size_t in_dim = mat.rows();
        size_t out_dim = mat.cols();
        const double fan_in = static_cast<double>(in_dim);
        const double fan_out = static_cast<double>(out_dim);
        switch(dist){
            case Distribution::Normal:
                std_dev = gain;
                break;
            case Distribution::XavierNormal:
                std_dev = gain * std::sqrt(2.0 / (fan_in + fan_out));
                break;
            case Distribution::HeNormal:
                std_dev = gain * std::sqrt(2.0 / fan_in);
                break;
        }

        std::normal_distribution<double> distr(0.0, std_dev);
        mat = Matrix::NullaryExpr(mat.rows(), mat.cols(),
                                    [&]() { return distr(gen_); });
    }

    // probably just std::vector is enough
    template<std::ranges::random_access_range T> 
    void shuffle(T& to_shuffle) {
        std::shuffle(to_shuffle.begin(), to_shuffle.end(), gen_);
    }

    template<typename T>
    T get_random_value(std::uniform_real_distribution<T> dist) {
        return dist(gen_);
    }

private:
    std::mt19937 gen_;
    uint32_t seed_;
};
}  // namespace nns
