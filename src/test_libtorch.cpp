#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Hello LibTorch!\n";
    torch::Tensor t = torch::rand({ 2, 3 });
    std::cout << t << std::endl;
    return 0;
}
