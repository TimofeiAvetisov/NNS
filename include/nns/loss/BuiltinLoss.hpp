#pragma once

#include <stdexcept>
#include <string>

#include <nns/core/Types.hpp>

namespace nns {
inline void validate_same_nonempty_shape(const Matrix& y_hat, const Matrix& y,
                                         const char* context) {
    if (y_hat.rows() != y.rows() || y_hat.cols() != y.cols()) {
        throw std::invalid_argument(std::string(context) + ": y_hat and y shapes must match");
    }
    if (y_hat.rows() == 0 || y_hat.cols() == 0) {
        throw std::invalid_argument(std::string(context) + ": inputs must be non-empty");
    }
}

struct MSELoss {
    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "MSELoss::loss");
        const Scalar n = static_cast<Scalar>(y_hat.size());
        return (y_hat - y).squaredNorm() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "MSELoss::gradient");
        return (Scalar{2.0} / static_cast<Scalar>(y_hat.size())) * (y_hat - y);
    }
};

struct MAELoss {
    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "MAELoss::loss");
        const Scalar n = static_cast<Scalar>(y_hat.size());
        return (y_hat - y).cwiseAbs().sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "MAELoss::gradient");
        const Scalar n = static_cast<Scalar>(y_hat.size());
        return (y_hat - y).cwiseSign() / n;
    }
};

struct HuberLoss {
    Scalar delta = Scalar{1.0};

    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "HuberLoss::loss");
        if (delta <= Scalar{0.0}) {
            throw std::invalid_argument("HuberLoss::loss: delta must be greater than zero");
        }

        const Scalar n = static_cast<Scalar>(y_hat.size());
        auto diff = y_hat - y;
        auto abs_diff = diff.cwiseAbs();
        const Matrix loss = (abs_diff.array() <= delta)
                                .select(Scalar{0.5} * diff.array().square(),
                                        delta * (abs_diff.array() - Scalar{0.5} * delta));
        return loss.sum() / n;
    }
    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "HuberLoss::gradient");
        if (delta <= Scalar{0.0}) {
            throw std::invalid_argument("HuberLoss::gradient: delta must be greater than zero");
        }

        const Scalar n = static_cast<Scalar>(y_hat.size());
        const Matrix diff = y_hat - y;
        const Matrix abs_diff = diff.cwiseAbs();
        return (abs_diff.array() <= delta).select(diff.array(), delta * diff.cwiseSign().array()) /
               n;
    }
};

struct BCELoss {
    Scalar eps = Scalar{1e-12};

    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "BCELoss::loss");
        if (eps <= Scalar{0.0} || eps >= Scalar{0.5}) {
            throw std::invalid_argument("BCELoss::loss: eps must be in (0, 0.5)");
        }

        const Scalar n = static_cast<Scalar>(y_hat.size());
        const Matrix clipped = y_hat.cwiseMax(eps).cwiseMin(Scalar{1.0} - eps);
        return -(y.array() * clipped.array().log() +
                 (Scalar{1.0} - y.array()) * (Scalar{1.0} - clipped.array()).log())
                    .sum() /
               n;
    }
    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "BCELoss::gradient");
        if (eps <= Scalar{0.0} || eps >= Scalar{0.5}) {
            throw std::invalid_argument("BCELoss::gradient: eps must be in (0, 0.5)");
        }

        const Scalar n = static_cast<Scalar>(y_hat.size());
        const Matrix clipped = y_hat.cwiseMax(eps).cwiseMin(Scalar{1.0} - eps);
        return ((clipped - y).array() / (clipped.array() * (Scalar{1.0} - clipped.array())))
                   .matrix() /
               n;
    }
};

struct BCEWithLogitsLoss {  // for numerical stability with sigmoid inside the loss
    Scalar eps = Scalar{1e-12};

    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "BCEWithLogitsLoss::loss");
        const Scalar n = static_cast<Scalar>(y_hat.size());
        const Matrix max_zero = y_hat.cwiseMax(Scalar{0.0});
        const Matrix log_exp = (Scalar{1.0} + (-y_hat.array().abs()).exp()).log().matrix();
        return (max_zero - y_hat.cwiseProduct(y) + log_exp).sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "BCEWithLogitsLoss::gradient");
        const Scalar n = static_cast<Scalar>(y_hat.size());
        return (Scalar{1.0} / (Scalar{1.0} + (-y_hat.array()).exp()) - y.array()).matrix() / n;
    }
};

struct CrossEntropyLoss {  // using softmax + cross-entropy for numerical stab
    Scalar eps = Scalar{1e-12};

    static Matrix softmax(const Matrix& X) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("CrossEntropyLoss::softmax: X must be non-empty");
        }

        auto max_per_col = X.colwise().maxCoeff();
        Matrix shifted = X.rowwise() - max_per_col;
        Matrix expX = shifted.array().exp();
        auto sum_per_col = expX.colwise().sum();
        return expX.array().rowwise() / sum_per_col.array();
    }

    Scalar loss(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "CrossEntropyLoss::loss");
        const Scalar n = static_cast<Scalar>(y_hat.cols());
        Matrix probs = softmax(y_hat).cwiseMax(eps);
        return -(y.array() * probs.array().log()).sum() / n;
    }

    Matrix gradient(const Matrix& y_hat, const Matrix& y) const {
        validate_same_nonempty_shape(y_hat, y, "CrossEntropyLoss::gradient");
        const Scalar n = static_cast<Scalar>(y_hat.cols());
        return (softmax(y_hat) - y) / n;
    }
};
}  // namespace nns
