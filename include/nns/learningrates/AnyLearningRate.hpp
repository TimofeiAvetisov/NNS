#pragma once

#include <proxy/proxy.h>

namespace nns {
namespace LearningRatesProxy {
PRO_DEF_MEM_DISPATCH(MemGetLR, get_lr);
struct LearningRateScheduler : pro::facade_builder ::add_convention<MemGetLR, double(size_t) const>::build {};
} // namespace LearningRatesProxy

using AnyLearningRateScheduler = pro::proxy<LearningRatesProxy::LearningRateScheduler>;

template <typename T>
AnyLearningRateScheduler make_AnyLearningRateScheduler(T&& lr_scheduler) {
    return pro::make_proxy<LearningRatesProxy::LearningRateScheduler, T>(std::forward<T>(lr_scheduler));
}
} // namespace nns
