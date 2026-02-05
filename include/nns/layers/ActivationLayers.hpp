#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <stdexcept>
#include <memory>
#include <utility>

class ActivationLayer {
public:
    explicit ActivationLayer(AnyScalarActivation activation) : act_(std::move(activation)) {
        if (!act_.isDefined()) {
            throw std::invalid_argument("ActivationLayer constructor: activation is not defined");
        }
    }
    ActivationLayer(const ActivationLayer&) = delete;
    ActivationLayer& operator=(const ActivationLayer&) = delete;
    ActivationLayer(ActivationLayer&&) = default;
    ActivationLayer& operator=(ActivationLayer&&) = default;

    Matrix forward(Matrix X) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        cache_X_ = std::move(X);
        cache_Y_ = Y;
        return Y;
    }

    Matrix predict(const Matrix& X) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        return Y;
    }

    Matrix backward(Matrix dY, LinearGrads& /*grads*/) override {
        if ((dY.rows() != cache_Y_.rows()) || (dY.cols() != cache_Y_.cols())) {
            throw std::invalid_argument(
                "ActivationLayer::backward: dimension mismatch between dY and last_Y_");
        }

        for (int i = 0; i < dY.rows(); ++i) {
            for (int j = 0; j < dY.cols(); ++j) {
                double x = cache_X_(i, j);
                double y = cache_Y_(i, j);
                dY(i, j) *= act_->derivative(x, y);
            }
        }
        return dY;
    }

    LinearGrads form_grads() const {
        return LinearGrads(0, 0);
    }

    void sgd_step(double /*lr*/, LinearGrads& /*grads*/) {
        // Activation layer has no parameters, so nothing to do here
    }

private:
    AnyScalarActivation act_;
    Matrix cache_X_;
    Matrix cache_Y_;
};