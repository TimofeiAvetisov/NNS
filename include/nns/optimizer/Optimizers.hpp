#pragma once
#include <proxy/proxy.h>
#include <nns/core/OptCache.hpp>
#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/core/Data.hpp>
#include <nns/optimizer/BuiltinOptimizers.hpp>

namespace nns {

namespace OptimizerProxy {
PRO_DEF_MEM_DISPATCH(MemUpdate, update_weights);
PRO_DEF_MEM_DISPATCH(MemStep, step);

struct Optimizer : pro::facade_builder ::add_convention<MemUpdate, Data(Data, const LinearGrads&, OptCache&) const> ::add_convention<MemStep, void() const> ::build{};
}

using AnyOptimizer = pro::proxy<OptimizerProxy::Optimizer>;
template <typename T>
AnyOptimizer make_AnyOptimizer(T&& opt) {
    return pro::make_proxy<OptimizerProxy::Optimizer, T>(std::forward<T>(opt));
}
}