#pragma once
#include <torch/torch.h>

class LinearDataset : public torch::data::datasets::Dataset<LinearDataset> {
public:
    LinearDataset(size_t size = 1000) {
        
        data_ = torch::rand({ (long)size, 1 }) * 2 - 1;
        targets_ = 2 * data_ + 1;  // y = 2x + 1

    }

    torch::data::Example<> get(size_t index) override {
        return { data_[index], targets_[index] };
    }

    torch::optional<size_t> size() const override {
        return data_.size(0);
    }

private:
    torch::Tensor data_;
    torch::Tensor targets_;
};