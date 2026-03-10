#pragma once
#include <cstddef>

namespace nns {
template<typename T, typename Tag>
struct StrongType {
    explicit StrongType(T v) : value_(v) {}
    operator T() const {
        return value;
    }
    T value_;
};
}  // namespace nns