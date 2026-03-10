#pragma once
#include <nns/core/OptCache.hpp>
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Data.hpp>
#include <nns/learningrates/LearningRates.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>

// TODO need to rebuild logic, so Optimizer will call the Network by itself(in this way we can implement batch descents, otherwise logic is really hard)

namespace nns {

using BatchSize = StrongType<size_t, struct BatchSzTag>;

class SGD {
public:
    SGD() : lr_scheduler_(make_AnyLearningRateScheduler(ConstatLR())) {};

    template <typename T>
    SGD(T&& lr_scheduler, BatchSize batch_size = BatchSize{32})
        : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))),
          batch_size_(batch_size) {
    }

    Data update_weights(Data params, const LinearGrads& grads, OptCache& /*not used*/)
        const {  // basicly need to move Params into this method and then store the updated params
        if (grads.is_empty() || params.is_empty()) {
            return params;
        }
        auto [A, b] = decode_params(std::move(params));
        const Matrix& dA = std::any_cast<const Matrix&>(grads.get_dA().get_data());
        const Vector& db = std::any_cast<const Vector&>(grads.get_db().get_data());
        double lr = lr_scheduler_->get_lr(iter_);
        A -= lr * dA;
        b -= lr * db;
        return Data(std::make_pair(A, b));
    }

    void step() const {
        ++iter_;
    }

private:
    AnyLearningRateScheduler lr_scheduler_;
    size_t batch_size_;
    mutable size_t iter_ = 0;
};

using MomentumCoef = StrongType<double, struct MomentumTag>;

// store {prev_a_change, prev_b_change in cache}
class Momentum {
public:
    Momentum() : lr_scheduler_(make_AnyLearningRateScheduler(ConstatLR())) {
    }

    template <typename T>
    Momentum(T&& lr_scheduler = ConstantLR(), MomentumCoef alpha = MomentumCoef{0.9})
        : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))),
          alpha_(alpha) {
    }

    Data update_weights(Data params, const LinearGrads& grads, OptCache& OptCache) const {
        auto [A, b] = decode_params(std::move(params));
        const Matrix& dA = std::any_cast<const Matrix&>(grads.get_dA().get_data());
        const Vector& db = std::any_cast<const Vector&>(grads.get_db().get_data());
        const double lr = lr_scheduler_->get_lr(iter_);
        Matrix A_step;
        Vector b_step;
        if (OptCache.is_inited()) {
            const Matrix cached_A_step =
                std::any_cast<const Matrix>(OptCache.get_data()[0].get_data());
            const Vector cached_b_step =
                std::any_cast<const Vector>(OptCache.get_data()[1].get_data());
            A_step = alpha_ * cached_A_step + lr * dA;
            b_step = alpha_ * cached_b_step + lr * db;
        } else {
            A_step = lr * dA;
            b_step = lr * db;
        }

        A -= A_step;
        b -= b_step;
        OptCache.store({Data(A_step), Data(b_step)});
        return Data(std::make_pair(A, b));
    }

    void step() const {
        ++iter_;
    }

private:
    AnyLearningRateScheduler lr_scheduler_;
    mutable size_t iter_ = 0;
    double alpha_ = 0.9;
};

using Beta1 = StrongType<double, struct Beta1Tag>;
using Beta2 = StrongType<double, struct Beta2Tag>;
using Eps = StrongType<double, struct EpsTag>;

// store {A_mass, A_vel, b_mass, b_vel}
class Adam {
public:
    Adam()
        : lr_scheduler_(make_AnyLearningRateScheduler(ConstatLR())),
          beta1_(0.9),
          beta2_(0.999),
          eps_(1e-8) {
    }

    template <typename T>
    Adam(T&& lr_scheduler, BatchSize batch_size = BatchSize{32}, Beta1 beta1 = Beta1{0.9}, Beta2 beta2 = Beta2{0.999},
         Eps eps = Eps{1e-8})
        : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))),
          batch_size_(batch_size),
          beta1_(beta1),
          beta2_(beta2),
          eps_(eps) {
    }

    Data update_weights(Data params, const LinearGrads& grads, OptCache& OptCache) const {
        auto [A, b] = decode_params(std::move(params));
        const Matrix& dA = std::any_cast<const Matrix&>(grads.get_dA().get_data());
        const Vector& db = std::any_cast<const Vector&>(grads.get_db().get_data());
        const double lr = lr_scheduler_->get_lr(iter_);
        Matrix A_step, A_mass, A_vel;
        Vector b_step, b_mass, b_vel;
        if (OptCache.is_inited()) {
            A_mass = std::any_cast<Matrix>(OptCache.get_data()[0].get_data());
            A_vel = std::any_cast<Matrix>(OptCache.get_data()[1].get_data());
            b_mass = std::any_cast<Matrix>(OptCache.get_data()[2].get_data());
            b_vel = std::any_cast<Matrix>(OptCache.get_data()[3].get_data());
        } else {
            A_mass =
                Matrix::Zero(static_cast<Index>(dA.rows()), static_cast<Index>(dA.cols())).eval();
            A_vel =
                Matrix::Zero(static_cast<Index>(dA.rows()), static_cast<Index>(dA.cols())).eval();
            b_mass = Vector::Zero(static_cast<Index>(db.rows())).eval();
            b_vel = Vector::Zero(static_cast<Index>(db.rows())).eval();
        }

        A_mass = beta1_ * A_mass + (1 - beta1_) * dA;
        A_vel = beta2_ * A_vel + (1 - beta2_) * (dA * dA.transpose());
        b_mass = beta1_ * b_mass + (1 - beta1_) * db;
        b_vel = beta2_ * b_vel + (1 - beta2_) * (db * db.transpose());

        size_t k = iter_ + 1;
        Matrix A_mass_hat = A_mass / (1 - pow(beta1_, k));
        Matrix A_vel_hat = A_vel / (1 - pow(beta2_, k));
        Vector b_mass_hat = b_mass / (1 - pow(beta1_, k));
        Vector b_vel_hat = b_vel / (1 - pow(beta2_, k));

        A.array() -= (lr * A_mass_hat.array()) / (A_vel_hat.array().sqrt() + eps_);
        b.array() -= (lr * b_mass_hat.array()) / (b_vel_hat.array().sqrt() + eps_);

        OptCache.store({Data(A_mass), Data(A_vel), Data(b_mass), Data(b_vel)});
        return Data(std::make_pair(A, b));
    }

    void step() const {
        ++iter_;
    }

private:
    AnyLearningRateScheduler lr_scheduler_;
    size_t batch_size = 32;
    double beta1_;
    double beta2_;
    double eps_;
    mutable size_t iter_ = 0;
};

std::pair<Matrix, Vector> decode_params(Data&& params) {
    return std::any_cast<std::pair<Matrix, Vector>>(params.get_data());
}
}  // namespace nns
