#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <nns/core/Data.hpp>
#include <nns/core/OptCache.hpp>
#include <nns/optimizer/Optimizers.hpp>

#include <stdexcept>
#include <memory>
#include <utility>

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

    ActivationLayer(const ActivationLayer&) = delete;
    ActivationLayer& operator=(const ActivationLayer&) = delete;
    ActivationLayer(ActivationLayer&&) = default;
    ActivationLayer& operator=(ActivationLayer&&) = default;

    std::pair<Matrix, Cache> forward(Matrix X) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        return {Y, Cache(Data(std::move(X)), Data(Y))};
    }

    Matrix predict(Matrix X) {
        X = X.unaryExpr([this](double x) { return act_->forward(x); });
        return X;
    }

    std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) {
        const Matrix& cache_X = std::any_cast<const Matrix&>(cache.get_X().get_data());
        const Matrix& cache_Y = std::any_cast<const Matrix&>(cache.get_Y().get_data());
        if (dY.rows() != cache_Y.rows() || dY.cols() != cache_Y.cols()) {
            throw std::invalid_argument(
                "ActivationLayer::backward: dimension mismatch between dY and last_Y_");
        }

        dY = dY.cwiseProduct(cache_X.binaryExpr(
            cache_Y, [this](double x, double y) { return act_->derivative(x, y); }));
        return {dY, LinearGrads()};
    }

    void update(const LinearGrads&, const AnyOptimizer&, OptCache&) {
        // Activation layer has no parameters, so nothing to do here
    }

    LinearGrads zero_grads() const {
        return LinearGrads();
    }

private:
    AnyScalarActivation act_;
};
}  // namespace nns
