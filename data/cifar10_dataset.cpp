#include "cifar10_dataset.h"
#include "../common/utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

const std::vector<std::string> CIFAR10Dataset::CLASS_NAMES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};

CIFAR10Dataset::CIFAR10Dataset() : loaded(false)
{
}

CIFAR10Dataset::~CIFAR10Dataset()
{
}

bool CIFAR10Dataset::load_dataset(const std::string &data_path)
{
    if (loaded)
    {
        std::cout << "Dataset already loaded!" << std::endl;
        return true;
    }

    std::cout << "Loading CIFAR-10 dataset from: " << data_path << std::endl;

    // Load training batches
    for (int i = 1; i <= 5; ++i)
    {
        std::string batch_filename = data_path + "/data_batch_" + std::to_string(i) + ".bin";
        std::vector<Tensor4D> batch_images;
        std::vector<int> batch_labels;

        if (!read_binary_file(batch_filename, batch_images, batch_labels))
        {
            std::cerr << "Failed to load training batch " << i << std::endl;
            return false;
        }

        // for (auto &image : batch_images)
        // {
        //     std::cout << "Image shape: [" << image.batch_size << ", "
        //               << image.height << ", " << image.width << ", "
        //               << image.channels << "]" << std::endl;
        // }

        // Append to training data
        train_images.insert(train_images.end(), batch_images.begin(), batch_images.end());
        train_labels.insert(train_labels.end(), batch_labels.begin(), batch_labels.end());

        std::cout << "Loaded training batch " << i << ": " << batch_images.size() << " images" << std::endl;
    }

    // Load test batch
    std::string test_filename = data_path + "/test_batch.bin";
    if (!read_binary_file(test_filename, test_images, test_labels, true))
    {
        std::cerr << "Failed to load test batch" << std::endl;
        return false;
    }

    std::cout << "Loaded test batch: " << test_images.size() << " images" << std::endl;

    loaded = true;
    print_dataset_info();
    return true;
}

bool CIFAR10Dataset::read_binary_file(const std::string &filename,
                                      std::vector<Tensor4D> &images,
                                      std::vector<int> &labels,
                                      bool is_test_batch)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // CIFAR-10 format: each record is 3073 bytes (1 byte label + 3072 bytes image)
    const int record_size = 3073; // 1 + 32*32*3
    int num_records = file_size / record_size;
    num_records = 100; // testing limit
    std::cout << "Reading " << num_records << " records from " << filename << std::endl;

    // Read all data into buffer
    std::vector<uchar> buffer(file_size);
    file.read(reinterpret_cast<char *>(buffer.data()), file_size);
    file.close();

    // Parse records
    images.reserve(num_records);
    labels.reserve(num_records);

    for (int i = 0; i < num_records; ++i)
    {
        Tensor4D image(1, 32, 32, 3); // Single image tensor
        int label;

        parse_cifar_record(buffer, i * record_size, image, label);

        images.push_back(std::move(image));
        labels.push_back(label);
    }

    return true;
}

void CIFAR10Dataset::parse_cifar_record(const std::vector<uchar> &buffer, int offset,
                                        Tensor4D &image, int &label)
{
    // First byte is the label
    label = static_cast<int>(buffer[offset]);

    // Next 3072 bytes are the image data (32x32x3)
    // CIFAR-10 format: R channel, G channel, B channel (each 32x32)
    int pixel_idx = offset + 1;

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < 32; ++h)
        {
            for (int w = 0; w < 32; ++w)
            {
                //  std::cout << "Assigning pixel: c=" << c << " h=" << h << " w=" << w << " value=" << static_cast<int>(buffer[pixel_idx]) << std::endl;
                image(0, h, w, c) = static_cast<real_t>(buffer[pixel_idx]);
                pixel_idx++;
            }
        }
    }
}

void CIFAR10Dataset::normalize_data()
{
    std::cout << "Normalizing dataset..." << std::endl;

    for (auto &image : train_images)
    {
        utils::normalize_tensor(image);
    }

    for (auto &image : test_images)
    {
        utils::normalize_tensor(image);
    }

    std::cout << "Dataset normalized to [0, 1]" << std::endl;
}

void CIFAR10Dataset::shuffle_training_data()
{
    Random rng;

    // Create indices and shuffle them
    std::vector<int> indices(train_images.size());
    std::iota(indices.begin(), indices.end(), 0);
    rng.shuffle(indices);

    // Apply shuffling
    std::vector<Tensor4D> shuffled_images(train_images.size());
    std::vector<int> shuffled_labels(train_labels.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        shuffled_images[i] = std::move(train_images[indices[i]]);
        shuffled_labels[i] = train_labels[indices[i]];
    }

    train_images = std::move(shuffled_images);
    train_labels = std::move(shuffled_labels);
}

void CIFAR10Dataset::get_batch(int batch_idx, int batch_size, Tensor4D &batch_images,
                               std::vector<int> &batch_labels, bool is_training)
{
    const auto &images = is_training ? train_images : test_images;
    const auto &labels = is_training ? train_labels : test_labels;

    int start_idx = batch_idx * batch_size;
    int end_idx = std::min(start_idx + batch_size, static_cast<int>(images.size()));
    int actual_batch_size = end_idx - start_idx;

    // Initialize batch tensor
    batch_images.resize(actual_batch_size, 32, 32, 3);
    batch_labels.resize(actual_batch_size);

    // Copy data
    for (int i = 0; i < actual_batch_size; ++i)
    {
        const auto &src_image = images[start_idx + i];
        for (int h = 0; h < 32; ++h)
        {
            for (int w = 0; w < 32; ++w)
            {
                for (int c = 0; c < 3; ++c)
                {
                    batch_images(i, h, w, c) = src_image(0, h, w, c);
                }
            }
        }
        batch_labels[i] = labels[start_idx + i];
    }
}

int CIFAR10Dataset::get_num_batches(int batch_size, bool is_training) const
{
    size_t dataset_size = is_training ? train_images.size() : test_images.size();
    return (dataset_size + batch_size - 1) / batch_size; // Ceiling division
}

void CIFAR10Dataset::print_dataset_info() const
{
    std::cout << "\n=== CIFAR-10 Dataset Info ===" << std::endl;
    std::cout << "Training images: " << train_size() << std::endl;
    std::cout << "Test images: " << test_size() << std::endl;
    std::cout << "Image dimensions: 32x32x3" << std::endl;
    std::cout << "Number of classes: " << CLASS_NAMES.size() << std::endl;

    // Print class distribution for training set
    std::vector<int> class_counts(10, 0);
    for (int label : train_labels)
    {
        class_counts[label]++;
    }

    std::cout << "\nTraining set class distribution:" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "  " << i << " (" << CLASS_NAMES[i] << "): " << class_counts[i] << std::endl;
    }
    std::cout << "========================\n"
              << std::endl;
}
