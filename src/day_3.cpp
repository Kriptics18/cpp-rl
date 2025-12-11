#include <torch/torch.h>
#include "headers/mlp.h"
#include <iostream>
#include <fstream>

int day_3() {

    torch::Device device = torch::kCUDA;
    if (!torch::cuda::is_available()) {
		throw std::runtime_error("CUDA is not available. Exiting.");
    }

	torch::manual_seed(100101);

    // Model: 2 -> 64 -> 1
    MLP net(2, 64, 1);
    net->to(device);

    // Optimizer
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));

	std::ofstream csv("day_3_output.txt");
	csv << "iter, loss\n";

	const int num_iterations = rand() % 1000 + 1000;
    const int batch_size = 64;

    for (int iter = 0; iter < num_iterations; ++iter) {

        // Generate random data
        auto x = torch::rand({ batch_size, 2 }).to(torch::kCUDA);
        auto y = (x.select(1, 0).pow(2) + x.select(1, 1)).unsqueeze(1).to(torch::kCUDA);

        // Forward pass
        auto y_pred = net->forward(x);

        // Compute loss
        auto loss = torch::mse_loss(y_pred, y);

        // Backward pass and optimization step
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Log loss
        csv << iter << ", " << loss.item<double>() << "\n";

        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Loss: " << loss.item<double>() << "\n";
        }
    }


    csv.close();
    return 0;
}
