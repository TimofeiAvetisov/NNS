#pragma once
#include <Eigen/Dense>

namespace nns {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

template<typename Tag, typename T = double>
struct StrongType {
    explicit StrongType(T v) : value(v) {}
    T value;

    operator T() const {
        return value;
    }
};

}  // namespace nns
