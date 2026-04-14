#pragma once

#include <nns/core/Types.hpp>
#include <stdexcept>
namespace nns {
struct MSELoss {
    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = static_cast<double>(y_hat.size());
        return (y_hat - y).squaredNorm() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        return (2.0 / y_hat.size()) * (y_hat - y);
    }
};

struct MAELoss {
    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = static_cast<double>(y_hat.size());
        return (y_hat - y).cwiseAbs().sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        const double n = static_cast<double>(y_hat.size());
        return (y_hat - y).cwiseSign() / n;
    }
};

struct HuberLoss {
    double delta = 1.0;

    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        auto diff = y_hat - y;
        auto abs_diff = diff.cwiseAbs();
        const Matrix loss =
            (abs_diff.array() <= delta)
                .select(0.5 * diff.array().square(), delta * (abs_diff.array() - 0.5 * delta));
        return loss.sum() / n;
    }
    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        const Matrix diff = y_hat - y;
        const Matrix abs_diff = diff.cwiseAbs();
        return (abs_diff.array() <= delta).select(diff.array(), delta * diff.cwiseSign().array()) /
               n;
    }
};

struct BCELoss {
    double eps = 1e-12;

    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        const Matrix clipped = y_hat.cwiseMax(eps).cwiseMin(1.0 - eps);
        return -(y.array() * clipped.array().log() +
                 (1.0 - y.array()) * (1.0 - clipped.array()).log())
                    .sum() /
               n;
    }
    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        const Matrix clipped = y_hat.cwiseMax(eps).cwiseMin(1.0 - eps);
        return ((clipped - y).array() / (clipped.array() * (1.0 - clipped.array()))).matrix() / n;
    }
};

struct BCEWithLogitsLoss {  // for numerical stability with sigmoid inside the loss
    double eps = 1e-12;

    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        const Matrix max_zero = y_hat.cwiseMax(0.0);
        const Matrix log_exp = (1.0 + (-y_hat.array().abs()).exp()).log().matrix();
        return (max_zero - y_hat.cwiseProduct(y) + log_exp).sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.size();
        return (1.0 / (1.0 + (-y_hat.array()).exp()) - y.array()).matrix() / n;
    }
};

struct CrossEntropyLoss {  // using softmax + cross-entropy for numerical stab
    double eps = 1e-12;

    static Matrix softmax(const Matrix& X) {
        Matrix shifted = X.colwise() - X.colwise().maxCoeff().transpose();
        Matrix expX = shifted.array().exp();
        return expX.array().rowwise() / expX.colwise().sum().array();
    }

    double loss(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.cols();
        Matrix probs = softmax(y_hat).cwiseMax(eps);
        return -(y.array() * probs.array().log()).sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        const double n = y_hat.cols();
        return (softmax(y_hat) - y) / n;
    }
};
}  // namespace nns
