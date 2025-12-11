#include <iostream>
#include <fstream>
#include <filesystem>

#include <math.h>
#include <torch/torch.h>

#include "headers/mlp.h"
#include "headers/train_config.h"

namespace fs = std::filesystem;

int day_4(TrainingParamConfig config) {

    torch::Device device = torch::kCUDA;
    if (!torch::cuda::is_available()) {
		throw std::runtime_error("CUDA is not available. Exiting.");
    }

    if (!fs::is_directory("../logs")) {
        fs::create_directories("../logs");
	}

	torch::manual_seed(100101);

    // Model: 2 -> 64 -> 1
    MLP net(2, 64, 1);
    net->to(device);


	std::ofstream csv("../logs/day_4_output.txt");
	csv << "iter, loss\n";

    std::unique_ptr<torch::optim::Optimizer> optimizer;

    if (config.use_adam) {
        optimizer = std::make_unique<torch::optim::Adam>(
            net->parameters(), torch::optim::AdamOptions(config.lr));
    }
    else {
        optimizer = std::make_unique<torch::optim::SGD>(
            net->parameters(), torch::optim::SGDOptions(config.lr).momentum(0.9));
    }

    torch::optim::StepLR scheduler(
        *optimizer,
        /*step_size=*/config.steps,
        /*gamma=*/0.995
    );



    for (int iter = 0; iter < config.iter; ++iter) {

        // Generate random data
        auto x = torch::rand({ config.batch_size, 2 }).to(device);
        auto y = (x.select(1, 0).pow(2) + x.select(1, 1)).unsqueeze(1).to(device);

        // Forward pass
        auto y_pred = net->forward(x);

        // Compute loss
        auto loss = torch::mse_loss(y_pred, y);
        // Log loss
        csv << iter << ", " << loss.item<double>() << "\n";


        // Backward pass and optimization step
		optimizer->zero_grad();
        loss.backward();
		optimizer->step();
        scheduler.step();


        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Loss: " << loss.item<double>() << "\n";
        }
    }

    csv.close();

    torch::serialize::OutputArchive archive;
    net->save(archive);
    archive.save_to("../logs/model_day4.pt");

    
    MLP net_loaded(2, 64, 1);
    
    torch::serialize::InputArchive in_archive;
    in_archive.load_from("../logs/model_day4.pt");
    net_loaded->load(in_archive);

    auto test_pred = net_loaded->forward(torch::rand({ 1,2 }).to(device));



    return 0;
}
