#pragma once
#include <nns/core/Types.hpp>
#include <Eigen/Dense>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cmath>
namespace nns {
    enum class InitScheme {
        Normal,
        XavierNormal,
        HeNormal
    };

    class RandomGenerator {
    public:
        static RandomGenerator& instance() {
            static RandomGenerator inst;
            return inst;
        }

        RandomGenerator(const RandomGenerator&) = delete;
        RandomGenerator& operator=(const RandomGenerator&) = delete;
        RandomGenerator(RandomGenerator&&) = delete;
        RandomGenerator& operator=(RandomGenerator&&) = delete;

        void reseed(uint32_t seed) {
            gen_.seed(seed);
            seed_ = seed;
        }

        uint32_t seed() const noexcept {
            return seed_;
        }

        void show_seed() const {
            std::cout << "RandomGenerator seed: " << seed_ << std::endl;
        }

        Matrix init_linear_weights(
            size_t out_dim,
            size_t in_dim,
            InitScheme scheme = InitScheme::XavierNormal,
            double gain = 1.0
        ) {
            if (in_dim == 0 || out_dim == 0) {
                throw std::invalid_argument("init_linear_weights: zero dimension");
            }
            if (!(gain > 0.0)) {
                throw std::invalid_argument("init_linear_weights: gain must be > 0");
            }

            const double fan_in  = static_cast<double>(in_dim);
            const double fan_out = static_cast<double>(out_dim);

            double stddev = 1.0;
            switch (scheme) {
                case InitScheme::Normal:
                    stddev = gain;
                    break;
                case InitScheme::XavierNormal:
                    stddev = gain * std::sqrt(2.0 / (fan_in + fan_out));
                    break;
                case InitScheme::HeNormal:
                    stddev = gain * std::sqrt(2.0 / fan_in);
                    break;
                default:
                    throw std::logic_error("Unknown InitScheme");
            }

            return normal_matrix(out_dim, in_dim, 0.0, stddev);
        }

    private:
        RandomGenerator()
            : gen_(738547485u), seed_(738547485u) {}

        std::mt19937 gen_;
        uint32_t seed_;

        Matrix normal_matrix(
            size_t rows,
            size_t cols,
            double mean,
            double stddev
        ) {
            std::normal_distribution<double> dist(mean, stddev);
            return Matrix::NullaryExpr(
                static_cast<Eigen::Index>(rows),
                static_cast<Eigen::Index>(cols),
                [&]() { return dist(gen_); }
            );
        }
    };
}