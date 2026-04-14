#pragma once
#include <proxy/proxy.h>

#include <nns/core/Types.hpp>
#include <nns/loss/BuiltinLoss.hpp>

namespace nns {
namespace LossProxy {
PRO_DEF_MEM_DISPATCH(MemLoss, loss);
PRO_DEF_MEM_DISPATCH(MemGradient, gradient);

// clang-format off
struct LossFunction
    : pro::facade_builder
    ::add_convention<MemLoss, double(const Matrix&, const Matrix&) const>
    ::add_convention<MemGradient, Matrix(const Matrix&, const Matrix&) const>::build {};
}  // namespace LossProxy
// clang-format on

using AnyLossFunction = pro::proxy<LossProxy::LossFunction>;
template <typename T, typename... Args>
AnyLossFunction make_AnyLossFunction(Args&&... args) {
    return pro::make_proxy<LossProxy::LossFunction, T>(T{std::forward<Args>(args)...});
}

}  // namespace nns
