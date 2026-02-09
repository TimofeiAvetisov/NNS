#pragma once
#include <nns/core/Types.hpp>
#include <Eigen/Dense>

struct LinearGrads {
    Matrix dA;
    Vector db;

    LinearGrads(Index out_dim, Index in_dim)
        : dA(Matrix::Zero(out_dim, in_dim)), db(Vector::Zero(out_dim)) {
    }

    void set_zero() {
        dA.setZero();
        db.setZero();
    }
};