#pragma once
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <Random.hpp>
#include <memory>
#include <stdexcept>

class LinearLayer {
public:
    LinearLayer(const Matrix& A_init, const Vector& b_init) : A_(A_init), b_(b_init) {
        if (A_.rows() != b_.size()) {
            throw std::invalid_argument("LinearLayer constructor: incompatible dimensions");
        }
    }
    LinearLayer(size_t in_dim, size_t out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0) {

        A_ = RandomGenerator::instance().init_linear_weights(out_dim, in_dim, init_scheme, gain);
        b_ = Vector::Zero(static_cast<Eigen::Index>(out_dim));
    }

    Matrix forward(Matrix X) {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        cache_X_ = std::move(X);
        return Y;
    }

    Matrix predict(Matrix X) {
        X = A_ * X;
        X.colwise() += b_;
        return X;
    }

    Matrix backward(Matrix dY, LinearGrads grads) {
        grads.dA.noalias() += dY * cache_X_.transpose();
        grads.db.noalias() += dY.rowwise().sum();
        return A_.transpose() * dY;
    }

    void sgd_step(double lr, LinearGrads grads) {
        A_.noalias() -= lr * grads.dA;
        b_.noalias() -= lr * grads.db;
    }

    LinearGrads form_grads() const {
        return LinearGrads(A_.rows(), A_.cols());
    }

    // test purpose only
    int in_dim() const {
        return static_cast<int>(A_.cols());
    };
    int out_dim() const {
        return static_cast<int>(A_.rows());
    };

    const Matrix& get_weights() const {
        return A_;
    }
    const Vector& get_biases() const {
        return b_;
    }

    Matrix& weights_mut() {
        return A_;
    }
    Vector& biases_mut() {
        return b_;
    }

private:
    // weights and biases
    Matrix A_;        // shape (out_dim, in_dim)
    Vector b_;        // shape (out_dim)
    Matrix cache_X_;  // shape (in_dim, batch_size)
};