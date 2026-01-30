#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <iostream>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

enum class InitScheme {
    Normal,        // N(0, std)
    XavierNormal,  // N(0, sqrt(2/(fan_in+fan_out)) * gain)
    HeNormal       // N(0, sqrt(2/fan_in) * gain)
};
class RandomGenerator {
public:
    explicit RandomGenerator(uint32_t seed = 738547485u) : gen_(seed), seed_(seed) {
    }

    void reseed(uint32_t seed) {
        gen_.seed(seed);
        seed_ = seed;
    }

    void show_seed() const {
        std::cout << "RandomGenerator seed: " << seed_ << std::endl;
    }

    Matrix init_linear_weights(size_t out_dim, size_t in_dim,
                               InitScheme scheme = InitScheme::XavierNormal, double gain = 1.0) {
        if (!(gain > 0.0)) {
            throw std::invalid_argument("init_linear_weights: gain must be > 0");
        }

        const double fan_in = static_cast<double>(in_dim);
        const double fan_out = static_cast<double>(out_dim);

        switch (scheme) {
            case InitScheme::Normal: {
                return normal_matrix(out_dim, in_dim, 0.0, gain);  // idk probably useless
            }
            case InitScheme::XavierNormal: {
                const double std_dev = gain * std::sqrt(2.0 / (fan_in + fan_out));
                return normal_matrix(out_dim, in_dim, 0.0, std_dev);
            }
            case InitScheme::HeNormal: {
                const double std_dev = gain * std::sqrt(2.0 / fan_in);
                return normal_matrix(out_dim, in_dim, 0.0, std_dev);
            }
            default:
                throw std::invalid_argument("init_linear_weights: invalid scheme");
        }
    }

private:
    std::mt19937 gen_;
    uint32_t seed_;
    Matrix normal_matrix(size_t rows, size_t cols, double mean = 0.0, double stddev = 1.0) {
        std::normal_distribution<double> dist(mean, stddev);
        return Matrix::NullaryExpr(rows, cols, [&]() {
            return dist(gen_);
        });  // Basicly just functor with normal distribution
    }
};