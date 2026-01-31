#pragma once
#include <nns/core/Tape.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/core/Types.hpp>
#include <stdexcept>
#include <utility>

struct ActivationNode : public TapeNode {
    Matrix X_;  // cashed input
    Matrix Y_;  // cashed output
    const AnyScalarActivation* act_;

    ActivationNode(Matrix X, Matrix Y, const AnyScalarActivation& act)
        : X_(std::move(X)), Y_(std::move(Y)), act_(&act) {
    }

    Matrix backward(const Matrix& grad) override {
        if ((grad.rows() != Y_.rows()) || (grad.cols() != Y_.cols())) {
            throw std::invalid_argument(
                "ActivationLayer::backward: dimension mismatch between dY and last_Y_");
        }

        Matrix dX = grad;
        for (int i = 0; i < grad.rows(); ++i) {
            for (int j = 0; j < grad.cols(); ++j) {
                double x = X_(i, j);
                double y = Y_(i, j);
                dX(i, j) *= (*act_)->derivative(x, y);
            }
        }
        return dX;
    }
};