#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <thread>
#include <vector>
#include <random>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

void different_albedo(const string& output_path, const string& file_path, double gamma, double strength) {
    cout << "Work on image: " << file_path << endl;

    // if (fs::exists(output_path)) {
    //     cout << output_path << " exists" << endl;
    //     return;
    // }

    Mat img = imread(file_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Could not open or find the image" << endl;
        return;
    }

    int rows = img.rows;
    int cols = img.cols;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrX(0, cols - 1);
    uniform_int_distribution<> distrY(0, rows - 1);

    int centerX = distrX(gen);
    int centerY = distrY(gen);
    int radius = max({centerX, centerY, cols - centerX, rows - centerY});
    
    uniform_int_distribution<> distrColor(0.95,1);
    Vec3b lightColor(distrColor(gen), distrColor(gen), distrColor(gen));

    double light_sum = lightColor[0] + lightColor[1] + lightColor[2];
    double r = lightColor[0] / light_sum;
    double g = lightColor[1] / light_sum;
    double b = lightColor[2] / light_sum;
    vector<double> lightWeight = {r, g, b};
    
    // randomly define the weight of r g b of incoming light


    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double distance = pow(centerY - j, 2) + pow(centerX - i, 2);
            if (distance < radius * radius) {
                double result = strength * (1.0 - pow(sqrt(distance) / radius, gamma));
                for (int c = 0; c < 3; ++c) {
                    result *= 3*lightWeight[c];
                    int val = img.at<Vec3b>(i, j)[c] + static_cast<int>(result);
                    img.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(val);
                }
            }
        }
    }

    imwrite(output_path, img);
    cout << "Wrote an image to " << output_path << endl;
}

void processFolder(const string& path, const string& folder) {
    string folder_path = path + "/" + folder;
    string images_path = folder_path + "/images";
    string output_folder = folder_path + "/images_albedo";

    if (!fs::exists(output_folder)) {
        fs::create_directory(output_folder);
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_strength(50, 200);
    uniform_real_distribution<> dis_gamma(0.9, 2.5);

    vector<thread> threads;
    for (const auto& entry : fs::directory_iterator(images_path)) {
        string file_path = entry.path().string();
        string output_path = output_folder + "/different_albedo_" + entry.path().filename().string();
        double strength = dis_strength(gen) * ( 1);
        double gamma = dis_gamma(gen);

        threads.push_back(thread(different_albedo, output_path, file_path, gamma, strength));
    }

    for (auto& t : threads) {
        t.join();
    }

    cout << "Processing complete for folder " << folder_path << ". Output stored in " << output_folder << endl;
}

int main() {
    string base_path = "/root/autodl-tmp/eth3d_high_res_test";

    for (const auto& entry : fs::directory_iterator(base_path)) {
        if (fs::is_directory(entry)) {
            processFolder(base_path, entry.path().filename().string());
        }
    }

    return 0;
}
