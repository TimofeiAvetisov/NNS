#pragma once
#include <nns/core/Types.hpp>
#include <any>
#include <typeindex>

namespace nns {
class Data {
public:
    Data(std::any data) : data_(std::move(data)) {}

    Data () : data_(std::any{}) {}

    bool is_empty() const {
        return data_.type() == typeid(void);
    }

    const Matrix& as_matrix() const {
        return std::any_cast<const Matrix&>(data_);
    }

    const Vector& as_vector() const {
        return std::any_cast<const Vector&>(data_);
    }

    Matrix& as_matrix_mutable() {
        return std::any_cast<Matrix&>(data_);
    }

    Vector& as_vector_mutable() {
        return std::any_cast<Vector&>(data_);
    }

    void set_zero() {
        as_matrix_mutable().setZero();
        as_vector_mutable().setZero();
    }

    const std::any& get_data() const { // currently for vector<Cache> dk yet how to fix
        return data_;
    }

    Data operator[](size_t index) const {
        return std::any_cast<std::vector<Data>>(data_)[index];
    }

private:
    std::any data_;
};
} // namespace nns