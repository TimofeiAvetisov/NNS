#pragma once
#include <nns/core/Types.hpp>
#include <any>
#include <typeindex>

namespace nns {
class Data {
public:
    explicit Data(Matrix data) : data_(std::move(data)), type_(typeid(Matrix)) {}
    explicit Data(Vector data) : data_(std::move(data)), type_(typeid(Vector)) {}

    Data () : type_(typeid(void)) {}

    bool is_empty() const {
        return type_ == typeid(void);
    }
    
    bool is_matrix() const {
        return type_ == typeid(Matrix);
    }

    bool is_vector() const {
        return type_ == typeid(Vector);
    }

    const Matrix& as_matrix() const {
        if (!is_matrix()) {
            throw std::bad_any_cast();
        }
        return std::any_cast<const Matrix&>(data_);
    }

    const Vector& as_vector() const {
        if (!is_vector()) {
            throw std::bad_any_cast();
        }
        return std::any_cast<const Vector&>(data_);
    }

    Matrix& as_matrix_mutable() {
        if (!is_matrix()) {
            throw std::bad_any_cast();
        }
        return std::any_cast<Matrix&>(data_);
    }

    Vector& as_vector_mutable() {
        if (!is_vector()) {
            throw std::bad_any_cast();
        }
        return std::any_cast<Vector&>(data_);
    }

    Data zeros_like() const {
        if (is_matrix()) {
            Matrix zeros = Matrix::Zero(as_matrix().rows(), as_matrix().cols());
            return Data(zeros);
        } else if (is_vector()) {
            Vector zeros = Vector::Zero(as_vector().size());
            return Data(zeros);
        } else {
            throw std::runtime_error("Data type is not set");
        }
    }

private:
    std::any data_;
    std::type_index type_;
};
} // namespace nns