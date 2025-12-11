#include "headers/train_config.h"
#include<torch/torch.h>
extern int day_4(TrainingParamConfig config);

int main() {
    TrainingParamConfig config;

    config.use_adam = false;
    config.steps = 150;
    config.iter = rand() % 1000 + 1000;

    return day_4(config);
}
