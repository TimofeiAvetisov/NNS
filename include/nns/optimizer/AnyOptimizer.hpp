#pragma once

#include <nns/core/Types.hpp>
#include <nns/optimizer/BuiltinOptimizers.hpp>

#include <proxy/proxy.h>

namespace nns {

namespace OptimizerProxy {
PRO_DEF_MEM_DISPATCH(MemUpdate, update_weights);
PRO_DEF_MEM_DISPATCH(MemStep, step);

struct Optimizer : pro::facade_builder
                   ::add_convention<MemUpdate, std::any(Matrix&, Matrix&&, std::any&&),
                                               std::any(Vector&, Vector&&, std::any&&)> 
                   ::add_convention<MemStep, void()>
                   ::build{};
}

using AnyOptimizer = pro::proxy<OptimizerProxy::Optimizer>;
template <typename T>
AnyOptimizer make_AnyOptimizer(T&& opt) {
    return pro::make_proxy<OptimizerProxy::Optimizer, T>(std::forward<T>(opt));
}
}