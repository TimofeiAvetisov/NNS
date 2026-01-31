#pragma once

#include <nns/core/Types.hpp>
#include <stdexcept>

enum class LossType { MSE };

class LossLayers {
public:
    explicit LossLayers(LossType type) : type_(type) {
    }

    double forward(const Matrix& y_hat, const Matrix& y) const {
        switch (type_) {
            case LossType::MSE: {
                // MSE loss function
                const double n = static_cast<double>(y_hat.size());
                return (y_hat - y).squaredNorm() / n;
            }
            default:
                throw std::invalid_argument("Invalid loss type");
        }
    }
    Matrix backward(const Matrix& y_hat, const Matrix& y) const {
        switch (type_) {
            case LossType::MSE: {
                const double n = static_cast<double>(y_hat.size());
                return (2.0 / n) * (y_hat - y);
            }
            default:
                throw std::invalid_argument("Invalid loss type");
        }
    }

private:
    LossType type_;
};
