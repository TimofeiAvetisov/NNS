#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>

namespace nns {
class Parameters {
public:
    Parameters(Matrix m) : data_(Data(std::move(m))), grad_(data_.zeros_like()) {}
    Parameters(Vector v) : data_(Data(std::move(v))), grad_(data_.zeros_like()) {}

    const Data& data() const {
        return data_;
    }

    const Data& grad() const {
        return grad_;
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