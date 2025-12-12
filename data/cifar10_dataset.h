#ifndef CIFAR10_DATASET_H
#define CIFAR10_DATASET_H

#include "../common/types.h"
#include <string>
#include <vector>

class CIFAR10Dataset
{
private:
    std::vector<Tensor4D> train_images;
    std::vector<Tensor4D> test_images;
    std::vector<int> train_labels;
    std::vector<int> test_labels;

    bool loaded;

    // Helper functions for binary file parsing
    bool read_binary_file(const std::string &filename, std::vector<Tensor4D> &images,
                          std::vector<int> &labels);
    void parse_cifar_record(const std::vector<uchar> &buffer, int offset,
                            Tensor4D &image, int &label);

public:
    CIFAR10Dataset();
    ~CIFAR10Dataset();

    // Load CIFAR-10 dataset from binary files
    bool load_dataset(const std::string &data_path);

    // Get data access
    const std::vector<Tensor4D> &get_train_images() const { return train_images; }
    const std::vector<Tensor4D> &get_test_images() const { return test_images; }
    const std::vector<int> &get_train_labels() const { return train_labels; }
    const std::vector<int> &get_test_labels() const { return test_labels; }

    // Dataset statistics
    size_t train_size() const { return train_images.size(); }
    size_t test_size() const { return test_images.size(); }
    bool is_loaded() const { return loaded; }

    // Data processing
    void normalize_data();
    void shuffle_training_data();

    // Batch generation
    void get_batch(int batch_idx, int batch_size, Tensor4D &batch_images,
                   std::vector<int> &batch_labels, bool is_training = true);
    int get_num_batches(int batch_size, bool is_training = true) const;

    // Utility functions
    void print_dataset_info() const;
    void save_sample_images(const std::string &output_dir, int num_samples = 10) const;

    // Class names for CIFAR-10
    static const std::vector<std::string> CLASS_NAMES;
};

#endif // CIFAR10_DATASET_H