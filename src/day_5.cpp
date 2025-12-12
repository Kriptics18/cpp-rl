//#include <iostream>
//#include <fstream>
//#include <filesystem>
//
//#include <math.h>
//#include <torch/torch.h>
//
//#include "headers/dataset.h"
//#include "headers/mlp.h"
//#include "headers/train_config.h"
//
//namespace fs = std::filesystem;
//
//int day_5(TrainingParamConfig config) {
//    auto train_dataset = LinearDataset(1000)
//        .map(torch::data::transforms::Stack<>());
//
//    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
//        std::move(train_dataset),
//        torch::data::DataLoaderOptions()
//        .batch_size(32)
//        .workers(2)
//    );
//
//    torch::Device device = torch::kCUDA;
//    if (!torch::cuda::is_available()) {
//        throw std::runtime_error("CUDA is not available. Exiting.");
//    }
//
//    if (!fs::is_directory("../logs")) {
//        fs::create_directories("../logs");
//    }
//
//    MLP model(1, 16, 1);
//    model->to(device);
//    model->train();
//
//    std::ofstream csv("../logs/day_5_output.txt");
//    csv << "iter, loss\n";
//
//    std::unique_ptr<torch::optim::Optimizer> optimizer;
//
//    if (config.use_adam) {
//        optimizer = std::make_unique<torch::optim::Adam>(
//            model->parameters(), torch::optim::AdamOptions(config.lr));
//    }
//    else {
//        optimizer = std::make_unique<torch::optim::SGD>(
//            model->parameters(), torch::optim::SGDOptions(config.lr).momentum(0.9));
//    }
//
//    torch::optim::StepLR scheduler(
//        *optimizer,
//        /*step_size=*/config.steps,
//        /*gamma=*/0.95
//    );
//
//    for (int epoch = 0; epoch < config.iter; epoch++) {
//        double epoch_loss = 0.0;
//
//        for (auto& batch : *train_loader) {
//            auto x = batch.data.to(device);
//            auto y = batch.target.to(device);
//
//            optimizer->zero_grad();
//            auto pred = model->forward(x);
//            auto loss = torch::mse_loss(pred, y);
//            loss.backward();
//            optimizer->step();
//			scheduler.step();
//
//            epoch_loss += loss.item<double>();
//        }
//
//        std::cout << "Epoch " << epoch
//            << " | Loss: " << epoch_loss << "\n";
//    }
//
//    csv.close();
//
//    auto test_dataset = LinearDataset(200)
//        .map(torch::data::transforms::Stack<>());
//
//    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
//        std::move(test_dataset),
//        torch::data::DataLoaderOptions()
//        .batch_size(32)
//        .workers(2)
//    );
//
//    model->eval();
//    torch::NoGradGuard no_grad;
//
//    double val_loss = 0.0;
//    for (auto& batch : *test_loader) {
//        auto pred = model->forward(batch.data.to(device)).to(device);
//        auto loss = torch::mse_loss(pred, batch.target.to(device)).to(device);
//        val_loss += loss.item<double>();
//    }
//
//    std::cout << "Validation Loss: " << val_loss << "\n";
//
//	torch::save(model, "../logs/model_day5.pt");
//
//	MLP model_loaded(1, 16, 1);
//	torch::load(model_loaded, "../logs/model_day5.pt");
//
//}

#include <filesystem>
#include <math.h>

#include "headers/dataset.h"
#include "headers/stdtraintest.h"

namespace fs = std::filesystem;

int day_5(TrainingParamConfig config) {
    auto train_dataset = LinearDataset(1000)
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
        .batch_size(32)
        .workers(2)
    );

    torch::Device device = torch::kCUDA;
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available. Exiting."); // we only run on CUDA. i refuse to be a peasant. i paid for my 5080 goddammit, i will use it as much as I can oof
    }

    if (!fs::is_directory("../logs")) {
        fs::create_directories("../logs");
    }

    MLP model(1, 16, 1);
    model->to(device);

    std::unique_ptr<torch::optim::Optimizer> optimizer;
    if (config.use_adam) {
        optimizer = std::make_unique<torch::optim::Adam>(
            model->parameters(), torch::optim::AdamOptions(config.lr));
    }
    else {
        optimizer = std::make_unique<torch::optim::SGD>(
            model->parameters(), torch::optim::SGDOptions(config.lr).momentum(0.9));
    }

    torch::optim::StepLR scheduler(
        *optimizer,
        /*step_size=*/config.steps,
        /*gamma=*/0.95
    );

    // Training (moved to header)
    stdtraintest::train_model(model, train_loader, device, *optimizer, scheduler, config, "../logs/day_5_output.txt");

    auto test_dataset = LinearDataset(200)
        .map(torch::data::transforms::Stack<>());

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset),
        torch::data::DataLoaderOptions()
        .batch_size(32)
        .workers(2)
    );

    // Testing (moved to header)
    double val_loss = stdtraintest::test_model(model, test_loader, device);
    std::cout << "Validation Loss: " << val_loss << "\n";

    torch::save(model, "../logs/model_day5.pt");

    MLP model_loaded(1, 16, 1);
    torch::load(model_loaded, "../logs/model_day5.pt");

    return 0;
}