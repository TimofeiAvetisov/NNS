#pragma once

#include <nns/core/Types.hpp>
#include <nns/utils/Random.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace nns {

enum In : Index;
enum Out : Index;

class LinearLayer {
public:
    LinearLayer() = default;
    LinearLayer(In in_dim, Out out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0)
        : b_(Vector::Zero(out_dim)) {
        A_ = RandomGenerator::instance().init_linear_weights(out_dim, in_dim, init_scheme,
                                                             gain);  // rework the rng
    }

    std::pair<Matrix, std::any> forward(Matrix X) {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        return {std::move(Y), Cache{std::move(X)}};
    }

    Matrix predict(Matrix X) const {
        X = A_ * X;
        X.colwise() += b_;
        return X;
    }

    std::pair<Matrix, std::any> backward(Matrix&& dY, const std::any& cache) {
        const Matrix& cache_X = std::any_cast<const Cache&>(cache).X_;
        Matrix dA = (dY * cache_X.transpose()).eval();
        Matrix db = (dY.rowwise().sum()).eval();
        return {A_.transpose() * dY, std::make_pair(std::move(dA), std::move(db))};
    }

    std::any update(std::any&& grads, AnyOptimizer& opt, std::any&& opt_cache) {
        auto [dA, db] = std::any_cast<std::pair<Matrix, Vector>>(grads);
        auto [opt_cache_A, opt_cache_b] = std::any_cast<std::pair<std::any, std::any>>(opt_cache);
        std::any opt_cache_A_new = opt->update_weights(A_, std::move(dA), std::move(opt_cache_A));
        std::any opt_cache_b_new = opt->update_weights(b_, std::move(db), std::move(opt_cache_b));
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
