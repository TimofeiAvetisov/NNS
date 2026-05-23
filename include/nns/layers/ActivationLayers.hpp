#pragma once
#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <nns/activation/AnyScalarActivation.hpp>
#include <nns/core/Types.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>

namespace nns {
class ActivationLayer {
public:
    template <typename Activation>
    explicit ActivationLayer(Activation&& activation)
        : act_(make_AnyScalarActivation(std::forward<Activation>(activation))) {
        if (!act_.has_value()) {
            throw std::invalid_argument("ActivationLayer constructor: activation is not defined");
        }
    }

    std::pair<Matrix, std::any> forward(Matrix&& X) {
        validate_input(X, "forward");
        Matrix Y = X.unaryExpr([this](Scalar x) { return act_->forward(x); });
        return {std::move(Y), Cache{std::move(X)}};
    }

    Matrix predict(const Matrix& X) const {
        validate_input(X, "predict");
        Matrix Y = X.unaryExpr([this](Scalar x) { return act_->forward(x); });
        return Y;
    }

    std::pair<Matrix, std::any> backward(Matrix&& dY, const std::any& cache) {
        if (!cache.has_value()) {
            throw std::invalid_argument("ActivationLayer::backward: cache is empty");
        }

        const Matrix& cache_X = std::any_cast<const Cache&>(cache).X_;
        if (dY.rows() != cache_X.rows() || dY.cols() != cache_X.cols()) {
            throw std::invalid_argument(
                "ActivationLayer::backward: dY shape must match cached input shape");
        }

        dY = dY.binaryExpr(cache_X,
                           [this](Scalar dy, Scalar x) { return dy * act_->derivative(x); });
        return {dY, {}};
    }

    std::any update(std::any&&, AnyOptimizer&, std::any&&) {
        // Activation layer has no parameters, so nothing to do here
        return {};
    }

private:
    void validate_input(const Matrix& X, const char* method) const {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument(std::string("ActivationLayer::") + method +
                                        ": X must be non-empty");
        }
    }

    struct Cache {
        Matrix X_;
    };

    AnyScalarActivation act_;
};
}  // namespace nns
