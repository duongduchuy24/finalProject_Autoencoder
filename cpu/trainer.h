#ifndef TRAINER_H
#define TRAINER_H

#include "../common/types.h"
#include "../common/utils.h"
#include "../data/cifar10_dataset.h"
#include "autoencoder.h"
#include <vector>

class Trainer
{
private:
    //=========================================================================
    // CORE COMPONENTS
    //=========================================================================
    std::unique_ptr<CPUAutoencoder> model;   // Autoencoder model
    std::unique_ptr<CIFAR10Dataset> dataset; // CIFAR-10 data loader

    //=========================================================================
    // TRAINING CONFIGURATION
    //=========================================================================
    int batch_size;       // Number of samples per batch
    int num_epochs;       // Total training epochs
    real_t learning_rate; // SGD learning rate

    //=========================================================================
    // TRAINING METRICS
    //=========================================================================
    std::vector<real_t> train_losses; // Loss per epoch
    std::vector<real_t> epoch_times;  // Time per epoch (ms)
    Timer timer;                      // Performance timer

public:
    Trainer(int batch_size = 32, int epochs = 20, real_t lr = 0.001f);
    ~Trainer();

    // Initialize dataset and model
    bool initialize(const std::string &data_path);

    // Training functions
    void train();
    real_t train_epoch();
    real_t evaluate_model();

    // Feature extraction
    void extract_features(const std::string &output_path);

    // Model management
    void save_model(const std::string &filename);
    bool load_model(const std::string &filename);

    // Utility functions
    void print_training_summary() const;
    void save_sample_reconstructions(const std::string &output_dir, int num_samples = 5);

    // Getters
    const std::vector<real_t> &get_train_losses() const { return train_losses; }
    const std::vector<real_t> &get_epoch_times() const { return epoch_times; }
};

#endif // TRAINER_H