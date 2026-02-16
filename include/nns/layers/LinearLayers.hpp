#pragma once
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Parameters.hpp>
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
    LinearLayer() : A_(Parameter(Matrix())), b_(Parameter(Vector())) {};
    LinearLayer(IN in_dim, OUT out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0) {
        A_ = Parameter(RandomGenerator::instance().init_linear_weights(out_dim, in_dim, init_scheme, gain));
        b_ = Parameter((Vector::Zero(static_cast<Eigen::Index>(out_dim))).eval());
    }

    std::pair<Matrix, Cache> forward(Matrix X) {
        Matrix Y = A_.data_matrix() * X;
        Y.colwise() += b_.data_vector();
        return std::make_pair(Y, Cache(std::move(X)));
    }

    Matrix predict(Matrix X) {
        X = A_.data_matrix() * X;
        X.colwise() += b_.data_vector();
        return X;
    }

    Matrix backward(Matrix dY, Cache cache) {
        A_.add_grad((dY * cache.get_X().transpose()).eval());
        b_.add_grad((dY.rowwise().sum()).eval());
        return A_.data_matrix().transpose() * dY;
    }

    void sgd_step(double lr, LinearGrads grads) {
        A_.add_grad((-lr * grads.dA).eval());
        b_.add_grad((-lr * grads.db).eval());
    }

    Parameter* get_weights_param() {
        return &A_;
    }

    Parameter* get_biases_param() {
        return &b_;
    }

private:
    // weights and biases
    Parameter A_;        // shape (out_dim, in_dim)
    Parameter b_;        // shape (out_dim)
};
}  // namespace nns