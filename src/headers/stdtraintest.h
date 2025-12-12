#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "headers/mlp.h"
#include "headers/train_config.h"

namespace stdtraintest {

    template <typename DataLoaderPtr>
    void train_model(
        MLP& model,
        DataLoaderPtr& train_loader,
        torch::Device device,
        torch::optim::Optimizer& optimizer,
        torch::optim::StepLR& scheduler,
        const TrainingParamConfig& config,
        const std::string& csv_path = "../logs/day_5_output.txt")
    {
        std::ofstream csv(csv_path);
        csv << "iter, loss\n";

        model->train();

        for (int epoch = 0; epoch < config.iter; ++epoch) {
            double epoch_loss = 0.0;

            for (auto& batch : *train_loader) {
                auto x = batch.data.to(device);
                auto y = batch.target.to(device);

                optimizer.zero_grad();
                auto pred = model->forward(x);
                auto loss = torch::mse_loss(pred, y);
                loss.backward();
                optimizer.step();
                scheduler.step();

                epoch_loss += loss.item<double>();
            }

            csv << epoch << "," << epoch_loss << "\n";
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss << "\n";
        }

        csv.close();
    }

    template <typename DataLoaderPtr>
    double test_model(
        MLP& model,
        DataLoaderPtr& test_loader,
        torch::Device device)
    {
        model->eval();
        torch::NoGradGuard no_grad;

        double val_loss = 0.0;
        for (auto& batch : *test_loader) {
            auto pred = model->forward(batch.data.to(device)).to(device);
            auto loss = torch::mse_loss(pred, batch.target.to(device)).to(device);
            val_loss += loss.item<double>();
        }

        return val_loss;
    }

} // namespace stdtraintest