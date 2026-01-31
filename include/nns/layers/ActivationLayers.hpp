#pragma once
#include <nns/core/Types.hpp>
#include <nns/activation/ScalarActivation.hpp>
#include <nns/nodes/ActivationNode.hpp>
#include <nns/core/Tape.hpp>
#include <nns/grads/LinearGrads.hpp>
#include <stdexcept>
#include <memory>

class ActivationLayer{
public:
    explicit ActivationLayer(AnyScalarActivation activation) : act_(std::move(activation)) {
        if (!act_.isDefined()) {
            throw std::invalid_argument("ActivationLayer constructor: activation is not defined");
        }
    }
    ActivationLayer(const ActivationLayer&) = delete;
    ActivationLayer& operator=(const ActivationLayer&) = delete;
    ActivationLayer(ActivationLayer&&) = default;
    ActivationLayer& operator=(ActivationLayer&&) = default;

    Matrix forward(const Matrix& X, Tape& tape, LinearGrads* /*grads*/) {
        Matrix Y = X.unaryExpr([this](double x) { return act_->forward(x); });
        tape.push(std::make_unique<ActivationNode>(X, Y, act_));
        return Y;
    }

    void sgd_step(double /*lr*/, LinearGrads* /*grads*/) {
        // Activation layer has no parameters, so nothing to do here
    }

private:
    AnyScalarActivation act_;
};