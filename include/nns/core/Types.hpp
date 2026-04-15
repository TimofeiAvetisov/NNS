#pragma once
#include <Eigen/Dense>

namespace nns {

using Scalar = double;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Index = Eigen::Index;

template <typename Tag, typename T = Scalar>
struct StrongType {
    explicit StrongType(T v) : value(v) {
    }
    T value;

    operator T() const {
        return value;
    }
};

}  // namespace nns
