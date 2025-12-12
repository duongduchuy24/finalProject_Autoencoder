#include "gpu_trainer.h"
#include <iostream>
#include <iomanip>
#include <fstream>

GPUTrainer::GPUTrainer(int batch_size, int epochs, real_t lr)
    : batch_size(batch_size), num_epochs(epochs), learning_rate(lr)
{
    model = std::make_unique<GPUAutoencoder>(lr);
    dataset = std::make_unique<CIFAR10Dataset>();
}

GPUTrainer::~GPUTrainer() {
    // Smart pointers handle cleanup
}

bool GPUTrainer::initialize(const std::string& data_path) {
    std::cout << "Initializing GPU trainer..." << std::endl;

    // Load dataset
    if (!dataset->load_dataset(data_path)) {
        std::cerr << "Failed to load dataset from: " << data_path << std::endl;
        return false;
    }

    // Normalize data
    dataset->normalize_data();

    std::cout << "GPU Training parameters:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Epochs: " << num_epochs << std::endl;
    std::cout << "  Learning rate: " << learning_rate << std::endl;
    std::cout << "  Training samples: " << dataset->train_size() << std::endl;
    std::cout << "  Test samples: " << dataset->test_size() << std::endl;

    model->print_model_info();
    return true;
}

void GPUTrainer::train() {
    std::cout << "\n=== Starting GPU Training ===" << std::endl;

    train_losses.clear();
    epoch_times.clear();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        timer.start();

        real_t epoch_loss = train_epoch();

        double epoch_time = timer.elapsed();
        epoch_times.push_back(static_cast<real_t>(epoch_time));
        train_losses.push_back(epoch_loss);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << num_epochs
                  << " - Loss: " << epoch_loss
                  << " - Time: " << epoch_time << "ms" << std::endl;

        // Save model checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            std::string checkpoint_name = "gpu_checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
            save_model(checkpoint_name);
        }
    }

    std::cout << "\n=== GPU Training Completed ===" << std::endl;
    print_training_summary();
}

real_t GPUTrainer::train_epoch() {
    real_t total_loss = 0.0f;
    int num_batches = dataset->get_num_batches(batch_size, true);

    // Shuffle training data at the beginning of each epoch
    dataset->shuffle_training_data();

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        Tensor4D batch_images;
        std::vector<int> batch_labels; // Not used for autoencoder training

        dataset->get_batch(batch_idx, batch_size, batch_images, batch_labels, true);

        // Train step (input = target for autoencoder)
        real_t batch_loss = model->train_step(batch_images, batch_images);
        total_loss += batch_loss;

        // Print progress every 50 batches (more frequent for GPU)
        if ((batch_idx + 1) % 50 == 0 || batch_idx == num_batches - 1) {
            std::cout << "  Batch " << std::setw(4) << (batch_idx + 1)
                      << "/" << num_batches
                      << " - Loss: " << std::fixed << std::setprecision(6) << batch_loss << std::endl;
        }
    }

    return total_loss / num_batches;
}

real_t GPUTrainer::evaluate_model() {
    std::cout << "\nEvaluating GPU model on test set..." << std::endl;

    real_t total_loss = 0.0f;
    int num_batches = dataset->get_num_batches(batch_size, false);

    timer.start();

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        Tensor4D batch_images;
        std::vector<int> batch_labels;

        dataset->get_batch(batch_idx, batch_size, batch_images, batch_labels, false);

        real_t batch_loss = model->evaluate(batch_images, batch_images);
        total_loss += batch_loss;
    }

    double eval_time = timer.elapsed();
    real_t avg_loss = total_loss / num_batches;

    std::cout << "GPU Test Loss: " << std::fixed << std::setprecision(6) << avg_loss
              << " - Evaluation Time: " << eval_time << "ms" << std::endl;

    return avg_loss;
}

void GPUTrainer::extract_features(const std::string& output_path) {
    std::cout << "\nExtracting features using GPU..." << std::endl;

    timer.start();

    // Extract training features
    std::ofstream train_features_file(output_path + "/gpu_train_features.bin", std::ios::binary);
    std::ofstream train_labels_file(output_path + "/gpu_train_labels.bin", std::ios::binary);

    if (!train_features_file || !train_labels_file) {
        std::cerr << "Failed to create GPU feature files" << std::endl;
        return;
    }

    int train_batches = dataset->get_num_batches(batch_size, true);
    const int feature_size = 8 * 8 * 128; // 8192 features

    for (int batch_idx = 0; batch_idx < train_batches; ++batch_idx) {
        Tensor4D batch_images;
        std::vector<int> batch_labels;

        dataset->get_batch(batch_idx, batch_size, batch_images, batch_labels, true);

        // Extract features using GPU encoder
        Tensor4D encoded_features;
        model->encode(batch_images, encoded_features);

        // Save features and labels
        for (int i = 0; i < encoded_features.batch_size; ++i) {
            // Flatten the 8x8x128 tensor to 8192 features
            for (int h = 0; h < 8; ++h) {
                for (int w = 0; w < 8; ++w) {
                    for (int c = 0; c < 128; ++c) {
                        real_t feature_val = encoded_features(i, h, w, c);
                        train_features_file.write(reinterpret_cast<const char*>(&feature_val), sizeof(real_t));
                    }
                }
            }

            int label = batch_labels[i];
            train_labels_file.write(reinterpret_cast<const char*>(&label), sizeof(int));
        }

        if ((batch_idx + 1) % 25 == 0) {
            std::cout << "  Processed " << (batch_idx + 1) << "/" << train_batches << " training batches" << std::endl;
        }
    }

    train_features_file.close();
    train_labels_file.close();

    // Extract test features (similar process)
    std::ofstream test_features_file(output_path + "/gpu_test_features.bin", std::ios::binary);
    std::ofstream test_labels_file(output_path + "/gpu_test_labels.bin", std::ios::binary);

    int test_batches = dataset->get_num_batches(batch_size, false);

    for (int batch_idx = 0; batch_idx < test_batches; ++batch_idx) {
        Tensor4D batch_images;
        std::vector<int> batch_labels;

        dataset->get_batch(batch_idx, batch_size, batch_images, batch_labels, false);

        Tensor4D encoded_features;
        model->encode(batch_images, encoded_features);

        for (int i = 0; i < encoded_features.batch_size; ++i) {
            for (int h = 0; h < 8; ++h) {
                for (int w = 0; w < 8; ++w) {
                    for (int c = 0; c < 128; ++c) {
                        real_t feature_val = encoded_features(i, h, w, c);
                        test_features_file.write(reinterpret_cast<const char*>(&feature_val), sizeof(real_t));
                    }
                }
            }

            int label = batch_labels[i];
            test_labels_file.write(reinterpret_cast<const char*>(&label), sizeof(int));
        }
    }

    test_features_file.close();
    test_labels_file.close();

    double extraction_time = timer.elapsed();

    std::cout << "GPU feature extraction completed in " << extraction_time << "ms" << std::endl;
    std::cout << "GPU features saved to: " << output_path << std::endl;
    std::cout << "  Training features: " << dataset->train_size() << " x " << feature_size << std::endl;
    std::cout << "  Test features: " << dataset->test_size() << " x " << feature_size << std::endl;
}

void GPUTrainer::save_model(const std::string& filename) {
    model->save_weights(filename);
}

bool GPUTrainer::load_model(const std::string& filename) {
    return model->load_weights(filename);
}

void GPUTrainer::print_training_summary() const {
    if (train_losses.empty()) {
        std::cout << "No GPU training data available" << std::endl;
        return;
    }

    std::cout << "\n=== GPU Training Summary ===" << std::endl;

    real_t total_time = 0.0f;
    for (real_t time : epoch_times) {
        total_time += time;
    }

    real_t avg_time_per_epoch = total_time / epoch_times.size();
    real_t final_loss = train_losses.back();
    real_t initial_loss = train_losses.front();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total GPU training time: " << total_time << "ms ("
              << total_time / 1000.0f << " seconds)" << std::endl;
    std::cout << "Average time per epoch: " << avg_time_per_epoch << "ms" << std::endl;
    std::cout << "Initial loss: " << initial_loss << std::endl;
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Loss reduction: " << ((initial_loss - final_loss) / initial_loss * 100.0f) << "%" << std::endl;
    std::cout << "============================" << std::endl;
}