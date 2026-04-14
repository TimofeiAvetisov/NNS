#pragma once

#include <proxy/proxy.h>

#include <nns/core/Types.hpp>
#include <nns/optimizer/BuiltinOptimizers.hpp>

namespace nns {
namespace OptimizerProxy {
PRO_DEF_MEM_DISPATCH(MemUpdate, update_weights);
PRO_DEF_MEM_DISPATCH(MemIterStep, iter_step);

// clang-format off
struct Optimizer
    : pro::facade_builder
    ::add_convention<MemUpdate, std::any(Matrix&, const Matrix&, std::any&&),
                                std::any(Vector&, const Vector&, std::any&&)>
    ::add_convention<MemIterStep, void()>::build {};
}  // namespace OptimizerProxy
// clang-format on

using AnyOptimizer = pro::proxy<OptimizerProxy::Optimizer>;

template <typename T>
inline AnyOptimizer make_AnyOptimizer(T&& opt) {
    return pro::make_proxy<OptimizerProxy::Optimizer, T>(std::forward<T>(opt));
}
}  // namespace nns
