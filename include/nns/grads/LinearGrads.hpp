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

    LinearGrads(Data dA, Data db = Data())
        : dA(std::move(dA)), db(std::move(db)), is_empty_(false) {
    }

    bool is_empty() const {
        return is_empty_;
    }

    const Data& get_dA() const {
        return dA;
    }
    const Data& get_db() const {
        return db;
    }

    Data& get_dA() {
        return dA;
    }

    Data& get_db() {
        return db;
    }

    LinearGrads operator[](size_t index) const {
        try {
            return std::any_cast<std::vector<LinearGrads>>(dA.get_data())[index];
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("LinearGrads operator[]: data is not a vector of LinearGrads");
        }
        // return std::any_cast<std::vector<LinearGrads>>(dA.get_data())[index];
    }
};
}  // namespace nns