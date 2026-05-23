#pragma once
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nns/core/Types.hpp>
#include <nns/utils/Random.hpp>

namespace nns {

struct Batch {
    Matrix X;
    Matrix Y;
};

enum BatchSize : size_t;
enum Shuffle : bool;
// remove it later

class DataLoader {
public:
    DataLoader(Matrix X, Matrix Y, BatchSize batch_size, Shuffle shuffle = Shuffle{true})
        : X_(std::move(X)), Y_(std::move(Y)), batch_size_(batch_size), shuffle_(shuffle) {

        if (X_.cols() != Y_.cols()) {
            throw std::invalid_argument("DataLoader: X and Y must have the same number of samples");
        }
        if (batch_size_ == 0) {
            throw std::invalid_argument("DataLoader: batch_size must be greater than zero");
        }
        if (X_.cols() == 0) {
            throw std::invalid_argument("DataLoader: dataset must contain at least one sample");
        }

        const size_t n = static_cast<size_t>(X_.cols());
        indices_.resize(n);
        std::iota(indices_.begin(), indices_.end(), Index{0});
    }

    void reset_epoch() {
        if (shuffle_) {
            throw std::logic_error("DataLoader: reset_epoch() without RNG requires Shuffle{false}");
        }
        ++epoch_;
    }

    void reset_epoch(RandomGenerator& rng) {
        if (shuffle_) {
            rng.shuffle(indices_);
        }
        ++epoch_;
    }

    size_t num_batches() const {
        return static_cast<size_t>(X_.cols()) / batch_size_;
    }

    size_t batch_size() const {
        return batch_size_;
    }

    size_t size() const {
        return static_cast<size_t>(X_.cols());
    }

    Batch get_batch(size_t batch_idx) const {
        if (batch_idx >= num_batches()) {
            throw std::out_of_range("DataLoader: batch_idx is out of range");
        }

        const size_t start = batch_idx * batch_size_;
        const size_t end = std::min(start + batch_size_, indices_.size());
        const Index cols = static_cast<Index>(end - start);

        Matrix bX(X_.rows(), cols);
        Matrix bY(Y_.rows(), cols);
        for (Index j = 0; j < cols; ++j) {
            const auto src = static_cast<Index>(indices_[start + static_cast<size_t>(j)]);
            bX.col(j) = X_.col(src);
            bY.col(j) = Y_.col(src);
        }
        return {std::move(bX), std::move(bY)};
    }

    class Iterator {
    public:
        Iterator(const DataLoader* loader, size_t idx) : loader_(loader), idx_(idx) {
        }

        Batch operator*() const {
            return loader_->get_batch(idx_);
        }
        Iterator& operator++() {
            ++idx_;
            return *this;
        }
        bool operator!=(const Iterator& other) const {
            return idx_ != other.idx_;
        }

    private:
        const DataLoader* loader_;
        size_t idx_;
    };

    Iterator begin() const {
        return {this, 0};
    }
    Iterator end() const {
        return {this, num_batches()};
    }

private:
    Matrix X_;
    Matrix Y_;
    size_t batch_size_;
    bool shuffle_;
    size_t epoch_ = 0;
    std::vector<Index> indices_;
};

}  // namespace nns
