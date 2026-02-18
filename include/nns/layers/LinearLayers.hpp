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
        return std::make_pair(Y, Cache(Data(std::move(X))));
    }

    Matrix predict(Matrix X) {
        X = A_ * X;
        X.colwise() += b_;
        return X;
    }

    std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) {
        LinearGrads grads(Data(std::move(zero_matrix())), Data(std::move(zero_vector())));
        try {
            grads.dA.as_matrix_mutable() = (dY * cache.get_X().transpose()).eval();
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(std::string("LinearLayer backward: ") + e.what());
        }
        grads.db.as_vector_mutable() = (dY.rowwise().sum()).eval();
        return {A_.transpose() * dY, std::move(grads)};
    }

    void update(const LinearGrads& grads /*, Optimizer opt, OptCache opt_cache*/) {
        // Due to not implemented optimizers
        const double lr = 0.01;
        A_ -= lr * grads.dA.as_matrix();
        b_ -= lr * grads.db.as_vector();
    }

private:
    // weights and biases
    Matrix A_;        // shape (out_dim, in_dim)
    Vector b_;        // shape (out_dim)

    Matrix zero_matrix() const {
        return Matrix::Zero(static_cast<Index>(A_.rows()), static_cast<Index>(A_.cols())).eval();
    }

    Vector zero_vector() const {
        return Vector::Zero(static_cast<Index>(A_.size())).eval();
    }
};
}  // namespace nns