#pragma once
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <Random.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace nns {

enum IN : size_t;
enum OUT : size_t;

class LinearLayer {
public:
    LinearLayer() : A_(Matrix()), b_(Vector()) {};
    LinearLayer(IN in_dim, OUT out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0) {
        A_ = RandomGenerator::instance().init_linear_weights(out_dim, in_dim, init_scheme, gain);
        b_ = (Vector::Zero(static_cast<Eigen::Index>(out_dim)));
    }

    std::pair<Matrix, Cache> forward(Matrix X) {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        return std::make_pair(Y, Cache(std::move(X)));
    }

    Matrix predict(Matrix X) {
        X = A_ * X;
        X.colwise() += b_
        return X;
    }

    std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) {
        LinearGrads grads(static_cast<Index>(A_.rows()), static_cast<Index>(A_.cols()));
        grads.dA = (dY * cache.get_X().transpose()).eval();
        grads.db = (dY.rowwise().sum()).eval();
        return {A_.transpose() * dY, std::move(grads)};
    }

    void update(const LinearGrads& grads /*, Optimizer opt, OptCache opt_cache*/) {
        // Due to not implemented optimizers
        const double lr = 0.01;
        A_ -= lr * grads.dA;
        b_ -= lr * grads.db
    }

    LinearGrads zero_grads() const {
        return LinearGrads(static_cast<Index>(A_.rows()), static_cast<Index>(A_.cols()));
    }

private:
    // weights and biases
    Matrix A_;        // shape (out_dim, in_dim)
    Vector b_;        // shape (out_dim)
};
}  // namespace nns