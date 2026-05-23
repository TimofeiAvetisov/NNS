#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <nns/core/Types.hpp>

namespace nns {
struct LRTag {};
using LR = StrongType<LRTag>;

class ConstantLR {
public:
    ConstantLR() = default;
    ConstantLR(LR lr) : lr_(lr) {
        validate_lr(lr_);
    }

    Scalar get_lr() {
        validate_lr(lr_);
        return lr_;
    }

    void iter_step() {
        ++iter_;
    }

    size_t get_iter() const {
        return iter_;
    }

private:
    static void validate_lr(Scalar lr) {
        if (lr <= Scalar{0.0}) {
            throw std::invalid_argument("ConstantLR: learning rate must be greater than zero");
        }
    }

    size_t iter_ = 1;
    Scalar lr_ = Scalar{0.01};
};

class TimeDecayLR {
public:
    TimeDecayLR() = default;
    TimeDecayLR(LR lr) : lr_(lr) {
        validate_params();
    }
    Scalar get_lr() {
        validate_params();
        Scalar lr = lr_ * std::pow((s0 / (s0 + iter_)), p);
        return lr;
    }

    void iter_step() {
        ++iter_;
    }

    size_t get_iter() const {
        return iter_;
    }

private:
    void validate_params() const {
        if (lr_ <= Scalar{0.0}) {
            throw std::invalid_argument("TimeDecayLR: learning rate must be greater than zero");
        }
        if (s0 <= Scalar{0.0}) {
            throw std::invalid_argument("TimeDecayLR: s0 must be greater than zero");
        }
        if (p < Scalar{0.0}) {
            throw std::invalid_argument("TimeDecayLR: p must be non-negative");
        }
    }

    Scalar s0 = Scalar{1.0};
    Scalar p = Scalar{0.5};
    Scalar lr_ = Scalar{1.0};
    size_t iter_ = 1;
};
}  // namespace nns
