#include "headers/train_config.h"
#include<torch/torch.h>
extern int day_5(TrainingParamConfig config);

int main() {
    TrainingParamConfig config;

    config.use_adam = false;
    config.steps = 150;
	config.iter = 5;

    return day_5(config);
}
