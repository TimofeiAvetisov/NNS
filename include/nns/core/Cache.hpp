#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>

namespace nns {
class Cache {
public:
    Cache() : cache_X_(), cache_Y_() {}
    Cache(Matrix&& X, Matrix Y) : cache_X_(std::move(X)), cache_Y_(std::move(Y)) {}
    Cache(Matrix&& X) : cache_X_(std::move(X)), cache_Y_() {}

    const Matrix& get_X() const {
        return cache_X_;
    }

    const Matrix& get_Y() const {
        return cache_Y_;
    }

private:
    Matrix cache_X_;
    Matrix cache_Y_;
};
} // namespace nns