#pragma once

#include "gpu_autoencoder.h"
#include "../data/cifar10_dataset.h"
#include "../common/timer.h"
#include <memory>
#include <vector>

class GPUTrainer {
private:
    int batch_size;
    int num_epochs;
    real_t learning_rate;
    
    std::unique_ptr<GPUAutoencoder> model;
    std::unique_ptr<CIFAR10Dataset> dataset;
    
    Timer timer;
    std::vector<real_t> train_losses;
    std::vector<real_t> epoch_times;
    
public:
    GPUTrainer(int batch_size = 64, int epochs = 10, real_t lr = 0.001f);
    ~GPUTrainer();
    
    bool initialize(const std::string& data_path);
    void train();
    real_t train_epoch();
    real_t evaluate_model();
    void extract_features(const std::string& output_path);
    
    void save_model(const std::string& filename);
    bool load_model(const std::string& filename);
    
    void print_training_summary() const;
};