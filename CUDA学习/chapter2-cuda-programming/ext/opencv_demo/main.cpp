#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("data/deer.png");
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    cv::imshow("Loaded Image", image);
    cv::waitKey(0);
    return 0;
}
