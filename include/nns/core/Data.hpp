#pragma once
#include <nns/core/Types.hpp>
#include <any>
#include <typeindex>
#include <stdexcept>
#include <iostream>
#include <string>

namespace nns {
class Data {
public:
    Data(std::any data) : data_(std::move(data)) {
    }

    Data() : data_(std::any{}) {
    }

    Data(const Data& other) : data_(other.data_) {
    }

    Data(Data&& other) noexcept : data_(std::move(other.data_)) {
    }

    Data& operator=(const Data& other) {
        if (this != &other) {
            data_ = other.data_;
        }
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }

    bool is_empty() const {
        return data_.type() == typeid(void);
    }

    const std::any& get_data() const {  // currently for vector<Cache> dk yet how to fix
        return data_;
    }

    std::any& get_data() {
        return data_;
    }

    Data operator[](size_t index) const {
        try {
            return std::any_cast<std::vector<Data>>(data_)[index];
        } catch (const std::bad_any_cast& e) {
            throw std::runtime_error("Data operator[]: data is not a vector of Data");
        }
        // return std::any_cast<std::vector<Data>>(data_)[index];
    }

private:
    std::any data_;
};
}  // namespace nns