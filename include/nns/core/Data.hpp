#pragma once
#include <nns/core/Types.hpp>
#include <any>
#include <typeindex>
#include <stdexcept>
#include <iostream>
#include <string>

namespace nns {
class Data {
public:
    Data(std::any data) : data_(std::move(data)) {}

    Data () : data_(std::any{}) {}

    bool is_empty() const {
        return data_.type() == typeid(void);
    }

    const Matrix& as_matrix() const {
        try {
            return std::any_cast<const Matrix&>(data_);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data::as_matrix: data is not a Matrix");
        }
        // return std::any_cast<const Matrix&>(data_);
    }

    const Vector& as_vector() const {
        try {
            return std::any_cast<const Vector&>(data_);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data::as_vector: data is not a Vector");
        }
        // return std::any_cast<const Vector&>(data_);
    }

    Matrix& as_matrix_mutable() {
        try {
            return std::any_cast<Matrix&>(data_);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data::as_matrix_mutable: data is not a Matrix");
        }
        // return std::any_cast<Matrix&>(data_);
    }

    Vector& as_vector_mutable() {
        try {
            return std::any_cast<Vector&>(data_);
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data::as_vector_mutable: data is not a Vector");
        }
        // return std::any_cast<Vector&>(data_);
    }

    void set_zero() {
        as_matrix_mutable().setZero();
        as_vector_mutable().setZero();
    }

    const std::any& get_data() const { // currently for vector<Cache> dk yet how to fix
        return data_;
    }

    Data operator[](size_t index) const {
        try {
            return std::any_cast<std::vector<Data>>(data_)[index];
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data operator[]: data is not a vector of Data");
        }
        // return std::any_cast<std::vector<Data>>(data_)[index];
    }

private:
    std::any data_;
};
} // namespace nns