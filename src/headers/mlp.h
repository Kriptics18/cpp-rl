#pragma once
#include <torch/torch.h>

struct MLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };

    MLPImpl(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = torch::relu(fc1->forward(x));
        out = fc2->forward(out);
        return out;
    }

    void save(torch::serialize::OutputArchive& archive) const {
        archive.write("fc1_weight", fc1->weight);
        archive.write("fc1_bias", fc1->bias);

        archive.write("fc2_weight", fc2->weight);
        archive.write("fc2_bias", fc2->bias);
    }

    void load(torch::serialize::InputArchive& archive) {
        archive.read("fc1_weight", fc1->weight);
        archive.read("fc1_bias", fc1->bias);

        archive.read("fc2_weight", fc2->weight);
        archive.read("fc2_bias", fc2->bias);
    }


};
TORCH_MODULE(MLP);
