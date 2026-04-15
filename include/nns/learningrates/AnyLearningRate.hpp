#pragma once

#include <cstddef>
#include <utility>

#include <proxy/proxy.h>

#include <nns/core/Types.hpp>

namespace nns {
namespace LearningRatesProxy {
PRO_DEF_MEM_DISPATCH(MemGetLR, get_lr);
PRO_DEF_MEM_DISPATCH(MemIterStep, iter_step);
PRO_DEF_MEM_DISPATCH(MemGetIter, get_iter);

// clang-format off
struct LearningRateScheduler
    : pro::facade_builder
    ::add_convention<MemGetLR, Scalar()>
    ::add_convention<MemIterStep, void()>
    ::add_convention<MemGetIter, size_t() const>::build {};
}  // namespace LearningRatesProxy
// clang-format on

using AnyLearningRateScheduler = pro::proxy<LearningRatesProxy::LearningRateScheduler>;

template <typename T>
inline AnyLearningRateScheduler make_AnyLearningRateScheduler(T&& lr_scheduler) {
    return pro::make_proxy<LearningRatesProxy::LearningRateScheduler, T>(
        std::forward<T>(lr_scheduler));
}
}  // namespace nns
