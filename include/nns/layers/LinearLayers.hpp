#pragma once
#include <nns/core/Types.hpp>
#include <nns/nodes/LinearNode.hpp>
#include <nns/core/Tape.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <Random.hpp>
#include <memory>

class LinearLayer {
public:
    LinearLayer(const Matrix& A_init, const Vector& b_init) : A_(A_init), b_(b_init) {
        if (A_.rows() != b_.size()) {
            throw std::invalid_argument("LinearLayer constructor: incompatible dimensions");
        }
    }
    LinearLayer(size_t in_dim, size_t out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0, std::shared_ptr<RandomGenerator> rng) {
        if (!rng) {
            throw std::invalid_argument("LinearLayer constructor: rng is nullptr");
        }
        A_ = rng->init_linear_weights(out_dim, in_dim, init_scheme, gain);
        b_ = Vector::Zero(static_cast<int>(out_dim));
    }  // rng will be provided thru NeuralNetwork

    Matrix forward(const Matrix& X, Tape& tape, LinearGrads* grads) {
        if (!grads) {
            throw std::invalid_argument("LinearLayer::forward: grads is nullptr");
        }
        Matrix Y = A_ * X;
        Y.colwise() += b_;

        tape.push(std::make_unique<LinearNode>(X, A_, grads->dA, grads->db));
        return Y;
    }

    void sgd_step(double lr, LinearGrads* grads) {
        if (!grads) {
            throw std::invalid_argument("LinearLayer::sgd_step: grads is nullptr");
        }
        A_ -= lr * grads->dA;
        b_ -= lr * grads->db;
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
    Matrix A_;  // shape (out_dim, in_dim)
    Vector b_;  // shape (out_dim)
};