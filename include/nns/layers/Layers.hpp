#pragma once

#include <TypeErasure.hpp>
#include <concepts>
#include <memory>
#include <utility>

#include <nns/core/Types.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <nns/layers/LinearLayers.hpp>
#include <nns/layers/ActivationLayers.hpp>
#include <nns/core/Cache.hpp>

namespace nns {
struct ILayer {
    virtual std::pair<Matrix, Cache> forward(Matrix X) = 0;
    virtual Matrix predict(Matrix X) = 0;
    virtual std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) = 0;
    virtual void update(const LinearGrads& grads /*, Optimizer opt, OptCache opt_cache*/) = 0;  // ActivationLayer do nothing
    virtual LinearGrads zero_grads() const = 0;
    virtual ~ILayer() = default;
};

template <class T>
concept LayerLike = requires(T& t, Matrix X, const LinearGrads& g, Cache cache, const Cache& const_cache) {
    { t.forward(X) } -> std::same_as<std::pair<Matrix, Cache>>;
    { t.update(g) } -> std::same_as<void>;
    { t.predict(X) } -> std::same_as<Matrix>;
    { t.backward(X, const_cache) } -> std::same_as<std::pair<Matrix, LinearGrads>>;
    { t.zero_grads() } -> std::same_as<LinearGrads>;
};

template <class Base, class TObject>
class CLayerImpl : public Base {
    static_assert(LayerLike<TObject>,
                  "Layer must provide: "
                  "std::pair<Matrix, Cache> forward(Matrix), "
                  "void update(const LinearGrads&), "
                  "Matrix predict(Matrix), "
                  "std::pair<Matrix, LinearGrads> backward(Matrix, const Cache&), "
                  "LinearGrads zero_grads() const");

public:
    template <class U>
    explicit CLayerImpl(U&& obj) : object_(std::forward<U>(obj)) {
    }

    std::pair<Matrix, Cache> forward(Matrix X) override {
        return object_.forward(std::move(X));
    }

    Matrix predict(Matrix X) override {
        return object_.predict(std::move(X));
    }

    std::pair<Matrix, LinearGrads> backward(Matrix dY, const Cache& cache) override {

        return object_.backward(std::move(dY), cache);
    }
    
    void update(const LinearGrads& grads) override {
        object_.update(grads);
    }

private:
    TObject object_;
};

using AnyLayer = CAnyMovable<ILayer, CLayerImpl>;

inline AnyLayer MakeLinearLayer(IN in_dim, OUT out_dim,
                                InitScheme init_scheme = InitScheme::XavierNormal,
                                double gain = 1.0) {
    return AnyLayer(LinearLayer(in_dim, out_dim, init_scheme, gain));
}

inline AnyLayer MakeActivationLayer(AnyScalarActivation activation) {
    return AnyLayer(ActivationLayer(std::move(activation)));
}
}  // namespace nns
