#pragma once
#include <nns/core/Tape.hpp>
#include <nns/core/Types.hpp>
#include <utility>

struct LinearNode : public TapeNode {
    Matrix X_;         // cashed input
    const Matrix& A_;  // weights
    Matrix* dA_;       // gradient w.r.t. weights
    Vector* db_;       // gradient w.r.t. biases

    LinearNode(Matrix X, const Matrix& A, Matrix& dA, Vector& db)
        : X_(std::move(X)), A_(A), dA_(&dA), db_(&db) {
    }

    // forward pass we have Y = A * X + b
    void backward(Matrix& grad) override {
        (*dA_).noalias() += grad * X_.transpose();
        (*db_).noalias() += grad.rowwise().sum();
        grad = A_.transpose() * grad;
    }
};