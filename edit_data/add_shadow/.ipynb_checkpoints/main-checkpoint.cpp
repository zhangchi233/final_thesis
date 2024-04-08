#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <filesystem>
#include <future>
void addRandomShadow(const std::string &image_path, const std::string &output_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Image could not be loaded." << std::endl;
        return;
    }

    // Convert to BGRA if the image is not already in this format
    if (image.channels() < 4) {
        cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
    }

    int width = image.cols, height = image.rows;
    cv::Mat shadow = cv::Mat::zeros(height, width, CV_8UC1);

    // Random engine and distributions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> angle_dist(0, 2 * M_PI);
    std::uniform_real_distribution<> intensity_dist(0.2, 0.8);

    // Generate random angle and intensity
    double angle = angle_dist(gen);
    double intensity = intensity_dist(gen);
    double cos_angle = cos(angle), sin_angle = sin(angle);

    // Creating the gradient mask for shadow
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double distance = (x * cos_angle + y * sin_angle) / sqrt(width * width + height * height);
            distance = std::max(0.0, std::min(1.0, distance));
            shadow.at<uchar>(y, x) = static_cast<uchar>(255 * (1 - intensity * distance));
        }
    }

    cv::Mat shadowColor;
    cv::cvtColor(shadow, shadowColor, cv::COLOR_GRAY2BGRA);

    // Blend the original image and shadow
    cv::Mat shadowedImage;
    cv::addWeighted(image, 1.0, shadowColor, -0.5, 0.0, shadowedImage); // You can adjust the weight for different effects

    image = shadowedImage;
    cv::imwrite(output_path, image);
    std::cout << "Processing complete."+output_path << std::endl;
}


int main() {
    // File path
    std::string path = "/root/autodl-tmp/eth3d_high_res_test";
    std::random_device rd;
    std::mt19937 gen(rd()); // Declare 'gen' here
    // Iterate over folders
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_directory(entry)) {
            std::string folder = entry.path().string();

            // Vector to store futures
            std::vector<std::future<void>> futures;

            // Iterate over images in the folder
            for (const auto &img_entry : std::filesystem::directory_iterator(folder + "/images")) {
                std::string img_path = img_entry.path().string();
                std::string output_path = folder + "/images_shadow/different_dark_" +
                    std::filesystem::path(img_path).filename().string();

                // Random intensity
                std::uniform_real_distribution<> intensity_dis(0.3, 0.8);
                double intensity = intensity_dis(gen);

                // Submit tasks to the thread pool
                futures.push_back(std::async(std::launch::async, addRandomShadow, img_path, output_path));
            }

            // Wait for all tasks to complete
            for (auto &future : futures) {
                future.wait();
            }
        }
    }

    std::cout << "Processing complete." << std::endl;

    return 0;
}
