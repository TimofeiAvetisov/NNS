#pragma once

#include <TypeErasure.hpp>
#include <Eigen/Dense>
#include <concepts>
#include <functional>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct IScalarActivation {
    virtual double forward(double x) const = 0;
    virtual double derivative(double x, double y) const = 0;  // sometimes y is useful (e.g. for sigmoid)
    virtual ~IScalarActivation() = default;
};

template<class T>
concept ScalarActivationLike =
    requires(const T& t, const double x, const double y) {
        { t.forward(x) } -> std::same_as<double>;
        { t.derivative(x, y) } -> std::same_as<double>;
    };


template<class Base, class TObject>
class CScalarActivationImpl : public Base {
    static_assert(ScalarActivationLike<TObject>,
                  "Scalar activation must provide: double forward(double) const and double derivative(double) const");
public:
    template<class U>
    explicit CScalarActivationImpl(U&& obj) : object_(std::forward<U>(obj)) {}

    double forward(double x) const override {
        return object_.forward(x);
    }

    double derivative(double x) const override {
        return object_.derivative(x);
    }
private:
    TObject object_;
};

using AnyScalarActivation = CAnyMovable<IScalarActivation, CScalarActivationImpl>;


// Built-in activations

struct ReLU {
    double forward(double x) const {
        return x > 0.0 ? x : 0.0;
    }
    double derivative(double x, double /*y*/) const {
        return x > 0.0 ? 1.0 : 0.0;
    }
};

struct Sigmoid {
    double forward(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double derivative(double x, double y) const {
        return y * (1.0 - y);
    }
};

struct Tanh {
    double forward(double x) const {
        return std::tanh(x);
    }

    double derivative(double x, double y) const {
        return 1.0 - y * y;
    }

};


// Factory functions, need to think about it
inline AnyScalarActivation MakeReLU() {
    return AnyScalarActivation(ReLU{});
}

inline AnyScalarActivation MakeSigmoid() {
    return AnyScalarActivation(Sigmoid{});
}

inline AnyScalarActivation MakeTanh() {
    return AnyScalarActivation(Tanh{});
}

// i want to use initializer lists for network to construct, e.g. {{{10, 20}, {20, 10}}, {ReLu, Sigmoid}}

using ActivationFactory = std::function<AnyScalarActivation()>;

inline const ActivationFactory relu = [] {return MakeReLU(); };
inline const ActivationFactory sigmoid = [] {return MakeSigmoid(); };
inline const ActivationFactory tanh_act = [] {return MakeTanh(); };  // tanh is std function, so tanh_act



// To create own activations, e.g. LeakyReLU
/*
Create a function with forward and derivative methods, e.g.
struct LeakyReLU {
    double alpha;
    double forward(double x) const {
        return x > 0.0 ? x : alpha * x;
    }
    double derivative(double x, double y) const {
        return x > 0.0 ? 1.0 : alpha;
    }
};

Create a factory function, e.g.
inline const ActivationFactory leaky_relu(double alpha) {
    return [alpha] {return AnyScalarActivation(LeakyReLU{alpha}); };
}

and now you can pass leakyReLU like this: leaky_relu(0.01) to NeuroLink constructor
*/