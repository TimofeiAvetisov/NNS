#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <nns/core/Parameters.hpp>
#include <stdexcept>
#include <memory>
#include <utility>


namespace nns {
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

    std::pair<Matrix, Cache> forward(Matrix X) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        return std::make_pair(Y, Cache(std::move(X), Y));
    }

    Matrix predict(Matrix X) {
        X = X.unaryExpr([this](double x) { return act_->forward(x); });
        return X;
    }

    Matrix backward(Matrix dY, Cache cache) {
        const Matrix& cache_X = cache.get_X();
        const Matrix& cache_Y = cache.get_Y();
        if ((dY.rows() != cache_Y.rows()) || (dY.cols() != cache_Y.cols())) {
            throw std::invalid_argument(
                "ActivationLayer::backward: dimension mismatch between dY and last_Y_");
        }

        for (int i = 0; i < dY.rows(); ++i) {
            for (int j = 0; j < dY.cols(); ++j) {
                double x = cache_X(i, j);
                double y = cache_Y(i, j);
                dY(i, j) *= act_->derivative(x, y);
            }
        }
        return dY;
    }

    Parameter* get_weights_param() {
        return nullptr;  // Activation layer has no weights
    }

    Parameter* get_biases_param() {
        return nullptr;  // Activation layer has no biases
    }

    void sgd_step(double /*lr*/, LinearGrads /*grads*/) {
        // Activation layer has no parameters, so nothing to do here
    }

private:
    AnyScalarActivation act_;
};
}  // namespace nns