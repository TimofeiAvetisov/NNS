#pragma once
#include <any>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <ranges>
#include <stdexcept>

#include <Eigen/Dense>

#include <nns/core/Types.hpp>
namespace nns {
enum class Distribution { Normal, XavierNormal, HeNormal };

struct GainTag {};
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
        if (mat.rows() <= 0 || mat.cols() <= 0) {
            throw std::invalid_argument("RandomGenerator::init_matrix: matrix must be non-empty");
        }
        if (gain <= Scalar{0.0}) {
            throw std::invalid_argument(
                "RandomGenerator::init_matrix: gain must be greater than zero");
        }

        Scalar std_dev = Scalar{1.0};
        size_t in_dim = mat.rows();
        size_t out_dim = mat.cols();
        const Scalar fan_in = static_cast<Scalar>(in_dim);
        const Scalar fan_out = static_cast<Scalar>(out_dim);
        switch (dist) {
            case Distribution::Normal:
                std_dev = gain;
                break;
            case Distribution::XavierNormal:
                std_dev = gain * std::sqrt(Scalar{2.0} / (fan_in + fan_out));
                break;
            case Distribution::HeNormal:
                std_dev = gain * std::sqrt(Scalar{2.0} / fan_in);
                break;
        }

        std::normal_distribution<Scalar> distr(Scalar{0.0}, std_dev);
        mat = Matrix::NullaryExpr(mat.rows(), mat.cols(), [&]() { return distr(gen_); });
    }

    // probably just std::vector is enough
    template <std::ranges::random_access_range T>
    void shuffle(T& to_shuffle) {
        std::shuffle(to_shuffle.begin(), to_shuffle.end(), gen_);
    }

    template <typename T>
    T get_random_value(std::uniform_real_distribution<T> dist) {
        return dist(gen_);
    }

private:
    std::mt19937 gen_;
    uint32_t seed_;
};
}  // namespace nns
