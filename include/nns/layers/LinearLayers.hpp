#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <optional>

#include <nns/core/Types.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>
#include <nns/utils/Random.hpp>

namespace nns {
enum In : Index;
enum Out : Index;

class LinearLayer {
public:
    LinearLayer() = default;
    LinearLayer(In in_dim, Out out_dim, RandomGenerator& rng, Distribution dist = Distribution::Normal, Gain gain = Gain{1.0})
        : b_(Vector::Zero(out_dim)) {
        A_ = Matrix{out_dim, in_dim};
        rng.init_matrix(A_, dist, gain);
    }

    std::pair<Matrix, std::any> forward(Matrix&& X) {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        return {std::move(Y), Cache{std::move(X)}};
    }

    Matrix predict(const Matrix& X) const {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        return Y;
    }

    std::pair<Matrix, std::any> backward(Matrix&& dY, const std::any& cache) {
        const Matrix& cache_X = std::any_cast<const Cache&>(cache).X_;
        Matrix dA = (dY * cache_X.transpose()).eval();
        Vector db = (dY.rowwise().sum()).eval();
        return {A_.transpose() * dY, std::any(std::make_pair(std::move(dA), std::move(db)))};
    }

    std::any update(std::any&& grads, AnyOptimizer& opt, std::any&& opt_cache) {
        auto [dA, db] = std::any_cast<std::pair<Matrix, Vector>&&>(std::move(grads));
        std::any opt_cache_A, opt_cache_b;
        if (opt_cache.has_value()) {
            auto p = std::any_cast<std::pair<std::any, std::any>&&>(std::move(opt_cache));
            opt_cache_A = std::move(p.first);
            opt_cache_b = std::move(p.second);
        }
        std::any opt_cache_A_new = opt->update_weights(A_, dA, std::move(opt_cache_A));
        std::any opt_cache_b_new = opt->update_weights(b_, db, std::move(opt_cache_b));
        return std::make_pair(std::move(opt_cache_A_new), std::move(opt_cache_b_new));
    }

private:
    struct Cache {
        Matrix X_;
    };

    Matrix A_;
    Vector b_;
};
}  // namespace nns
