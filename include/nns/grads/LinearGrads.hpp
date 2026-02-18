#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>
#include <vector>
#include <Eigen/Dense>

namespace nns {
struct LinearGrads {
    Data dA;
    Data db;
    bool is_empty_;

    LinearGrads() : is_empty_(true) {
    }

    LinearGrads(Index out_dim, Index in_dim)
        : dA(Matrix::Zero(out_dim, in_dim)), db(Vector::Zero(out_dim)), is_empty_(false) {
    }

    LinearGrads(Data dA, Data db = Data()) : dA(std::move(dA)), db(std::move(db)), is_empty_(false) {
    }

    void set_zero() {
        dA.set_zero();
        db.set_zero();
    }
    // std::vector<LinearGrads>

    LinearGrads operator[](size_t index) const {
        return std::any_cast<std::vector<LinearGrads>>(dA.get_data())[index];
    }
};
}  // namespace nns