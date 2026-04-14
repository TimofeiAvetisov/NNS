#pragma once
#include <cmath>
#include <cstddef>

namespace nns {
class ConstatLR {
public:
    ConstatLR() = default;
    ConstatLR(double lr) : lr_(lr) {
    }

    double get_lr(size_t /*iter*/) const {
        return lr_;
    }

private:
    double lr_ = 0.01;
};

class TimeDecayLR {
public:
    double get_lr(size_t iter) const {
        return lr_ * std::pow((s0 / (s0 + iter)), p);
    }

private:
    double s0 = 1.0;
    double p = 0.5;
    double lr_ = 1.0;
};

}  // namespace nns
