#pragma once
#include <cmath>
#include <cstddef>

namespace nns {
struct LRTag{};
using LR = StrongType<LRTag>;

class ConstantLR {
public:
    ConstantLR() = default;
    ConstantLR(LR lr) : lr_(lr) {
    }

    double get_lr() {
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
    double lr_ = 0.01;
};

class TimeDecayLR {
public:
    TimeDecayLR() = default;
    TimeDecayLR(LR lr) : lr_(lr) {
    }
    double get_lr() {
        double lr = lr_ * std::pow((s0 / (s0 + iter_)), p);
        return lr;
    }

    void iter_step() {
        ++iter_;
    }

    size_t get_iter() const {
        return iter_;
    }

private:
    double s0 = 1.0;
    double p = 0.5;
    double lr_ = 1.0;
    size_t iter_ = 1;
};
}  // namespace nns
