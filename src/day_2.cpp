#include <torch/torch.h>
#include "headers/mlp.h"
#include <iostream>

int day_2() {
    MLP net(4, 64, 2);

    torch::Tensor x = torch::rand({ 1, 4 }); // batch size 1, input 4
    auto out = net->forward(x);

    std::cout << "Input:\n" << x << "\n";
    std::cout << "Output:\n" << out << "\n";

    return 0;
}
