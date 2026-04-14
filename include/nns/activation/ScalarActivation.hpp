#pragma once
#include <proxy/proxy.h>
#include <nns/core/Types.hpp>

namespace nns {
namespace ScalarActivationProxy {
PRO_DEF_MEM_DISPATCH(MemDerivative, derivative);
PRO_DEF_MEM_DISPATCH(MemForward, forward);

struct ScalarActivation
    : pro::facade_builder ::add_convention<MemForward, double(double) const>::add_convention<
          MemDerivative, double(double) const>::build {};

}  // namespace ScalarActivationProxy
using AnyScalarActivation = pro::proxy<ScalarActivationProxy::ScalarActivation>;

template <typename T>
AnyScalarActivation make_AnyScalarActivation(T&& activation) {
    return pro::make_proxy<ScalarActivationProxy::ScalarActivation, T>(std::forward<T>(activation));
}
}  // namespace nns