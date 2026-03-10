#pragma once
#include <nns/core/Types.hpp>
#include <nns/core/Data.hpp>

namespace nns {

class OptCache {
public:
    OptCache() : data_(Data()) {
    }

    OptCache(Data data) : data_(std::move(data)) {
    }

    const Data& get_data() const {
        return data_;
    }
private:
    Data data_;  // basicly it stores the vector<Data> for optimizer or just one Data, opt knows it anyway.
};
}  // namespace nns