#pragma once

#include <any>
#include <iostream>
#include <utility>
#include <vector>

#include <nns/core/Types.hpp>
#include <nns/loss/AnyLossFunction.hpp>
#include <nns/network/Network.hpp>
#include <nns/utils/DataLoader.hpp>
#include <nns/utils/Random.hpp>

namespace nns {
class Trainer {
public:
    Trainer(NeuralNetwork& net, AnyOptimizer& opt, AnyLossFunction& loss_func, DataLoader& data)
        : net_(net), opt_(opt), loss_func_(loss_func), data_(data) {
    }

    Scalar fit_epoch(RandomGenerator& rng) {
        data_.reset_epoch(rng);
        return train_current_epoch(true);
    }

    Scalar fit_epoch() {
        data_.reset_epoch();
        return train_current_epoch(false);
    }

    std::vector<Scalar> fit(size_t epochs, RandomGenerator& rng) {
        std::vector<Scalar> loss_history;
        loss_history.reserve(epochs);
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            loss_history.push_back(fit_epoch(rng));
        }
        return loss_history;
    }

    std::vector<Scalar> fit(size_t epochs) {
        std::vector<Scalar> loss_history;
        loss_history.reserve(epochs);
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            loss_history.push_back(fit_epoch());
        }
        return loss_history;
    }

private:
    Scalar train_current_epoch(bool show_progress) {
        Scalar loss_sum = Scalar{0.0};
        size_t batch_ind = 0;
        for (auto [bX, bY] : data_) {
            ++batch_ind;
            auto [pred_Y, layer_cache] = net_.forward(std::move(bX));
            Scalar loss_value = loss_func_->loss(pred_Y, bY);
            loss_sum += loss_value;
            Matrix dL_dy = loss_func_->gradient(pred_Y, bY);
            auto [dL_dx, gradients] = net_.backward(std::move(dL_dy), layer_cache);
            opt_cache_ = net_.update(std::move(gradients), opt_, std::move(opt_cache_));
            opt_->iter_step();

            if (show_progress) {
                std::cout << "batch: " << batch_ind << '\r';
            }
        }

        if (show_progress) {
            std::cout << '\n';
        }

        return loss_sum / static_cast<Scalar>(data_.num_batches());
    }

    NeuralNetwork& net_;
    AnyOptimizer& opt_;
    AnyLossFunction& loss_func_;
    DataLoader& data_;
    std::any opt_cache_{};
};
}  // namespace nns
