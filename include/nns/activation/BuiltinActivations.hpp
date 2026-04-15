#pragma once

#include <cmath>
#include <functional>

#include <nns/activation/AnyScalarActivation.hpp>

namespace nns {
struct ReLU {
    Scalar forward(Scalar x) const {
        return x > Scalar{0.0} ? x : Scalar{0.0};
    }
    Scalar derivative(Scalar x) const {
        return x > Scalar{0.0} ? Scalar{1.0} : Scalar{0.0};
    }
};

struct Sigmoid {
    Scalar forward(Scalar x) const {
        return Scalar{1.0} / (Scalar{1.0} + std::exp(-x));
    }

    Scalar derivative(Scalar x) const {
        Scalar y = forward(x);
        return y * (Scalar{1.0} - y);
    }
};

struct Tanh {
    Scalar forward(Scalar x) const {
        return std::tanh(x);
    }

    Scalar derivative(Scalar x) const {
        Scalar y = forward(x);
        return Scalar{1.0} - y * y;
    }
};
}  // namespace nns
