#pragma once

#include <proxy/proxy.h>

#include <nns/core/Types.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/layers/ActivationLayers.hpp>
#include <nns/optimizer/AnyOptimizer.hpp>

#include <any>


namespace nns {
namespace LayerProxy {
PRO_DEF_MEM_DISPATCH(MemForward, forward);
PRO_DEF_MEM_DISPATCH(MemPredict, predict);
PRO_DEF_MEM_DISPATCH(MemBackward, backward);
PRO_DEF_MEM_DISPATCH(MemUpdate, update);


struct Layer : pro::facade_builder 
               ::add_convention<MemForward, std::pair<Matrix, std::any>(Matrix)>
               ::add_convention<MemPredict, Matrix(Matrix) const>
               ::add_convention<MemBackward, std::pair<Matrix, std::any>(Matrix, const std::any&)>
               ::add_convention<MemUpdate, std::any(std::any&&, AnyOptimizer&, std::any&&)>::build {};
}  // namespace LayerProxy

using AnyLayer = pro::proxy<LayerProxy::Layer>;

AnyLayer make_AnyLayer(LinearLayer&& layer) {
    return pro::make_proxy<LayerProxy::Layer, LinearLayer>(std::move(layer));
}

template <typename Activation>
AnyLayer make_AnyLayer(Activation&& act) {
    return pro::make_proxy<LayerProxy::Layer, ActivationLayer>(std::forward<Activation>(act));
}
}  // namespace nns
