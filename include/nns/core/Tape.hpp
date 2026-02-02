#pragma once
#include <nns/core/Types.hpp>
#include <vector>
#include <memory>
#include <utility>

struct TapeNode {
    virtual ~TapeNode() = default;
    virtual void backward(Matrix& grad_output) = 0;
};

struct Tape {
    std::vector<std::unique_ptr<TapeNode>> nodes;

    void push(std::unique_ptr<TapeNode> node) {
        nodes.push_back(std::move(node));
    }

    void backward(Matrix& grad) {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            (*it)->backward(grad);
        }
        nodes.clear();
    }

    void clear() {
        nodes.clear();
    }
};

using TapePtr = std::shared_ptr<Tape>;
