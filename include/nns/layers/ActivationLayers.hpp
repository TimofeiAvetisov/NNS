#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>

#include <stdexcept>
#include <memory>
#include <utility>
#include <any>

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

    std::pair<Matrix, std::any> forward(Matrix X) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        return {std::move(Y), Cache{std::move(X)}};
    }

    Matrix predict(Matrix X) const {
        X = X.unaryExpr([this](double x) { return act_->forward(x); });
        return X;
    }

    std::pair<Matrix, std::any> backward(Matrix&& dY, const std::any& cache) {
        const Matrix& cache_X = std::any_cast<const Cache&>(cache).X_;
        dY = dY.binaryExpr(cache_X, [this](double dy, double x) { return dy * act_->derivative(x); });
        return {dY, {}};
    }

    void update(std::any&&, AnyOptimizer&, std::any&&) {
        // Activation layer has no parameters, so nothing to do here
    }

private:
    struct Cache {
        Matrix X_;
    };

    AnyScalarActivation act_;
};
}  // namespace nns
