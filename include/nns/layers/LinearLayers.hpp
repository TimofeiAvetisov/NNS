#pragma once

#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Cache.hpp>
#include <nns/utils/Random.hpp>
#include <nns/core/OptCache.hpp>
#include <nns/optimizer/Optimizers.hpp>
#include <nns/core/Data.hpp>
#include <nns/core/StrongType.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace nns {

using IN = StrongType<size_t, struct InTag>;
using OUT = StrongType<size_t, struct OutTag>;

class LinearLayer {
public:
    LinearLayer() = default;
    LinearLayer(IN in_dim, OUT out_dim, InitScheme init_scheme = InitScheme::XavierNormal,
                double gain = 1.0) {
        A_ = RandomGenerator::instance().init_linear_weights(out_dim, in_dim, init_scheme, gain);
        b_ = (Vector::Zero(static_cast<Eigen::Index>(out_dim)));
    }

    std::pair<Matrix, Cache> forward(Matrix X) {
        Matrix Y = A_ * X;
        Y.colwise() += b_;
        return {Y, Cache(Data(std::move(X)))};
    }

    Matrix predict(Matrix X) {
        X = A_ * X;
        X.colwise() += b_;
        return X;
    }

    std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) {
        LinearGrads grads(Data(std::move(zero_matrix())), Data(std::move(zero_vector())));
        const Data& data_x = cache.get_X();
        const Matrix& cache_X = std::any_cast<const Matrix&>(data_x.get_data());
        std::any_cast<Matrix&>(grads.get_dA().get_data()) = (dY * cache_X.transpose()).eval();
        std::any_cast<Vector&>(grads.get_db().get_data()) = (dY.rowwise().sum()).eval();
        return {A_.transpose() * dY, std::move(grads)};
    }

    void update(const LinearGrads& grads, const AnyOptimizer& opt, OptCache& opt_cache) {  // weights and biases not valid during opt::update_weights
        Data params(std::make_pair(std::move(A_), std::move(b_)));
        Data updated_weights = opt->update_weights(std::move(params), grads, opt_cache);
        auto& [new_A, new_b] = std::any_cast<std::pair<Matrix, Vector>&>(updated_weights.get_data());
        A_ = std::move(new_A);
        b_ = std::move(new_b);
    }

private:
    // weights and biases
    Matrix A_;  // shape (out_dim, in_dim)
    Vector b_;  // shape (out_dim)

    Matrix zero_matrix() const {
        return Matrix::Zero(static_cast<Index>(A_.rows()), static_cast<Index>(A_.cols())).eval();
    }

    Vector zero_vector() const {
        return Vector::Zero(static_cast<Index>(A_.rows())).eval();
    }
};
}  // namespace nns
