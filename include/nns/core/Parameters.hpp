#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>

namespace nns {
class Parameter {
public:
    Parameter() : data_(Data()), grad_(Data()) {}
    Parameter(Matrix m) : data_(Data(std::move(m))), grad_(data_.zeros_like()) {}
    Parameter(Vector v) : data_(Data(std::move(v))), grad_(data_.zeros_like()) {}

    const Data& data() const {
        return data_;
    }
    
    const Data& grad() const {
        return grad_;
    }

    Data& data_mut() {
        return data_;
    }

    Data& grad_mut() {
        return grad_;
    }

    const Matrix& data_matrix() const {
        if (!data_.is_matrix()) {
            throw std::runtime_error("Data is not a matrix");
        }
        return data_.as_matrix();
    }

    const Vector& data_vector() const {
        if (!data_.is_vector()) {
            throw std::runtime_error("Data is not a vector");
        }
        return data_.as_vector();
    }

    const Matrix& grad_matrix() const {
        if (!grad_.is_matrix()) {
            throw std::runtime_error("Gradient is not a matrix");
        }
        return grad_.as_matrix();
    }

    const Vector& grad_vector() const {
        if (!grad_.is_vector()) {
            throw std::runtime_error("Gradient is not a vector");
        }
        return grad_.as_vector();
    }


    void zero_grad() {
        grad_ = data_.zeros_like();
    }

    void add_grad(const Matrix& grad) {
        if (!grad_.is_matrix()) {
            throw std::runtime_error("Gradient is not a matrix");
        }
        grad_.as_matrix_mutable() += grad;
    }

    void add_grad(const Vector& grad) {
        if (!grad_.is_vector()) {
            throw std::runtime_error("Gradient is not a vector");
        }
        grad_.as_vector_mutable() += grad;
    }
private:
    Data data_;
    Data grad_;
};
} // namespace nns