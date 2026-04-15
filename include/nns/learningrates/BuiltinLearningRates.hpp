#pragma once
#include <cmath>
#include <cstddef>

#include <nns/core/Types.hpp>

namespace nns {
struct LRTag {};
using LR = StrongType<LRTag>;

class ConstantLR {
public:
    ConstantLR() = default;
    ConstantLR(LR lr) : lr_(lr) {
    }

    Scalar get_lr() {
        return lr_;
    }

    void iter_step() {
        ++iter_;
    }

    size_t get_iter() const {
        return iter_;
    }

private:
    size_t iter_ = 1;
    Scalar lr_ = Scalar{0.01};
};

class TimeDecayLR {
public:
    TimeDecayLR() = default;
    TimeDecayLR(LR lr) : lr_(lr) {
    }
    Scalar get_lr() {
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
    Scalar s0 = Scalar{1.0};
    Scalar p = Scalar{0.5};
    Scalar lr_ = Scalar{1.0};
    size_t iter_ = 1;
};
}  // namespace nns
