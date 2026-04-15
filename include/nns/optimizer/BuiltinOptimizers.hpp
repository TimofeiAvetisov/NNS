#pragma once
#include <any>
#include <cmath>
#include <unordered_map>

#include <nns/core/Types.hpp>
#include <nns/learningrates/AnyLearningRate.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>

namespace nns {
/// SGD optimizer
class SGDOptimizer {
public:
    SGDOptimizer() : lr_scheduler_(make_AnyLearningRateScheduler(ConstantLR())) {
    }

    template <typename T>
    explicit SGDOptimizer(T&& lr_scheduler)
        : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))){
    }

    std::any update_weights(Matrix& param, const Matrix& grad, std::any&&) {
        update_any(param, grad);
        return {};
    }

    std::any update_weights(Vector& param, const Vector& grad, std::any&&) {
        update_any(param, grad);
        return {};
    }

    void iter_step() {
        lr_scheduler_->iter_step();
    }

private:
    template <typename DataType>
    void update_any(DataType& param, const DataType& grad) {
        const double lr = lr_scheduler_->get_lr();
        param -= lr * grad;
    }

    AnyLearningRateScheduler lr_scheduler_;
};


struct Beta1Tag{};
struct Beta2Tag{};
struct EpsTag{};

using Beta1 = StrongType<Beta1Tag>;
using Beta2 = StrongType<Beta2Tag>;
using Eps = StrongType<EpsTag>;

/// Adam optimizer.
class AdamOptimizer {
public:
    AdamOptimizer() : lr_scheduler_(make_AnyLearningRateScheduler(ConstantLR(LR{0.001}))) {
    }

    template <typename T>
    explicit AdamOptimizer(T&& lr_scheduler, Beta1 beta1 = 0.9, Beta2 beta2 = 0.999,
                           Eps eps = 1e-8)
        : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps) {
    }

    std::any update_weights(Matrix& param, const Matrix& grad, std::any&& opt_cache) {
        Cache<Matrix> cache;
        if (!opt_cache.has_value()) {
            cache.m = Matrix::Zero(grad.rows(), grad.cols());
            cache.v = Matrix::Zero(grad.rows(), grad.cols());
        } else {
            cache = std::any_cast<Cache<Matrix>&&>(std::move(opt_cache));
        }
        adam_step(param, grad, cache);
        return cache;
    }

    std::any update_weights(Vector& param, const Vector& grad, std::any&& opt_cache) {
        Cache<Vector> cache;
        if (!opt_cache.has_value()) {
            cache.m = Vector::Zero(grad.size());
            cache.v = Vector::Zero(grad.size());
        } else {
            cache = std::any_cast<Cache<Vector>&&>(std::move(opt_cache));
        }
        adam_step(param, grad, cache);
        return cache;
    }

    void iter_step() {
        lr_scheduler_->iter_step();
    }

private:
    template <typename T>
    struct Cache {
        T m;
        T v;
    };

    template <typename T>
    void adam_step(T& param, const T& grad, Cache<T>& cache) {
        const double lr = lr_scheduler_->get_lr();
        const double t = lr_scheduler_->get_iter() + 1;

        cache.m = beta1_ * cache.m + (1.0 - beta1_) * grad;
        cache.v = beta2_ * cache.v + (1.0 - beta2_) * grad.cwiseProduct(grad);

        const T m_hat = cache.m / (1.0 - std::pow(beta1_, t));
        const T v_hat = cache.v / (1.0 - std::pow(beta2_, t));

        param.array() -= lr * m_hat.array() / (v_hat.array().sqrt() + eps_);
    }

    AnyLearningRateScheduler lr_scheduler_;
    double beta1_ = 0.9;
    double beta2_ = 0.999;
    double eps_ = 1e-8;
};

}  // namespace nns
