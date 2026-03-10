#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>

namespace nns {

class OptCache {
private:
    using VecData = std::vector<Data>;
    VecData data_;
    bool is_inited_ = false;
public:
    OptCache() : data_(VecData()) {
    }

    OptCache(VecData&& data) : data_(std::move(data)), is_inited_(true) {
    }

    const VecData& get_data() const {
        return data_;
    }

    VecData& get_data() {
        return data_;
    }

    void store(VecData&& data) {
        data_ = data;
        is_inited_ = true;
    }

    bool is_inited() const {
        return is_inited_;
    }
};
}  // namespace nns