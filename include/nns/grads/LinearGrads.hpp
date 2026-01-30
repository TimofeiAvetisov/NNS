#pragma once
#include <nns/core/Types.hpp>
#include <Eigen/Dense>

struct LinearGrads {
    Matrix dA;
    Vector db;

    LinearGrads(size_t out_dim, size_t in_dim)
        : dA(Matrix::Zero(static_cast<int>(out_dim), static_cast<int>(in_dim))),
          db(Vector::Zero(static_cast<int>(out_dim))) {
    }

    void set_zero() {
        dA.setZero();
        db.setZero();
    }
};