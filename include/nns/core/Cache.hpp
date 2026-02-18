#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>
#include <any>

namespace nns {
class Cache {
public:
    Cache() : cache_X_(), cache_Y_() {}
    Cache(Data&& cache_X) : cache_X_(std::move(cache_X)), cache_Y_() {}
    Cache(Data&& cache_X, Data cache_Y) : cache_X_(std::move(cache_X)), cache_Y_(std::move(cache_Y)) {}

    const Matrix& get_X() const {
        return cache_X_.as_matrix();
    }

    const Matrix& get_Y() const {
        return cache_Y_.as_matrix();
    }

    Cache operator[](size_t index) const {
        return std::any_cast<std::vector<Cache>>(cache_X_.get_data())[index];
    }

private:
    Data cache_X_;
    Data cache_Y_;
};
} // namespace nns