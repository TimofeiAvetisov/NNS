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
#include <nns/core/Parameters.hpp>

namespace nns {
struct ILayer {
    virtual std::pair<Matrix, Cache> forward(Matrix X) = 0;
    virtual Matrix predict(Matrix X) = 0;
    virtual Matrix backward(Matrix dY, Cache cache) = 0;
    virtual void sgd_step(double lr, LinearGrads grads) = 0;  // ActivationLayer do nothing
    virtual Parameter* get_weights_param() = 0;
    virtual Parameter* get_biases_param() = 0;
    virtual ~ILayer() = default;
};

template <class T>
concept LayerLike = requires(T& t, Matrix X, LinearGrads g, double lr, Cache cache) {
    { t.forward(X) } -> std::same_as<std::pair<Matrix, Cache>>;
    { t.sgd_step(lr, g) } -> std::same_as<void>;
    { t.predict(X) } -> std::same_as<Matrix>;
    { t.backward(X, cache) } -> std::same_as<Matrix>;
    { t.get_weights_param() } -> std::same_as<Parameter*>;
    { t.get_biases_param() } -> std::same_as<Parameter*>;
};

template <class Base, class TObject>
class CLayerImpl : public Base {
    static_assert(LayerLike<TObject>,
                  "Layer must provide: "
                  "Matrix forward(Matrix), "
                  "void sgd_step(double, LinearGrads), "
                  "Matrix predict(Matrix), "
                  "Matrix backward(Matrix, Cache), "
                  "Parameter* get_weights_param(), "
                  "Parameter* get_biases_param()");

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

    Matrix backward(Matrix dY, Cache cache) override {

        return object_.backward(std::move(dY), cache);
    }

    Parameter* get_weights_param() override {
        return object_.get_weights_param();
    }

    Parameter* get_biases_param() override {
        return object_.get_biases_param();
    }

    void sgd_step(double lr, LinearGrads grads) override {
        object_.sgd_step(lr, std::move(grads));
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
