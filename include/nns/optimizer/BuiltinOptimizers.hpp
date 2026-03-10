#pragma once
#include <nns/core/OptCache.hpp>
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Data.hpp>
#include <nns/learningrates/LearningRates.hpp>
#include <nns/learningrates/BuiltinLearningRates.hpp>

namespace nns {
class SGDOptimizer {
public:
    SGDOptimizer() : lr_scheduler_(make_AnyLearningRateScheduler(ConstatLR())) {};

    template<typename T>
    SGDOptimizer(T&& lr_scheduler) : lr_scheduler_(make_AnyLearningRateScheduler(std::forward<T>(lr_scheduler))) {}

    Data update_weights(Data params, const LinearGrads& grads, OptCache& /*not used*/) const {  // basicly need to move Params into this method and then store the updated params
        if (grads.is_empty() || params.is_empty()) {
            return params;
        }
        auto [A, b] = std::any_cast<std::pair<Matrix, Vector>&>(params.get_data());
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
    // const double lr = 0.01;
    AnyLearningRateScheduler lr_scheduler_;
    mutable size_t iter_ = 0;
};

class GOpt {
public:
    Data update_weights(Data params, const LinearGrads& grads, OptCache& /*not used*/) const {
    return params;
    }
    void step() const {
    }
};
}
