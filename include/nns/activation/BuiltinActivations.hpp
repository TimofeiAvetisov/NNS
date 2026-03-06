#pragma once
#include <nns/activation/ScalarActivation.hpp>
#include <functional>
#include <cmath>
namespace nns {
struct ReLU {
    double forward(double x) const {
        return x > 0.0 ? x : 0.0;
    }
    double derivative(double x, double /*y*/) const {
        return x > 0.0 ? 1.0 : 0.0;
    }
};

struct Sigmoid {
    double forward(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double /*x*/, double y) const {
        return y * (1.0 - y);
    }
};

struct Tanh {
    double forward(double x) const {
        return std::tanh(x);
    }

    double derivative(double x, double y) const {
        return 1.0 - y * y;
    }
};
}  // namespace nns