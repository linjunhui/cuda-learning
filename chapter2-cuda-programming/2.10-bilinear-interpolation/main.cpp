#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("./images/1.png");
    cv::imshow("image", img);
    cv::waitKey();
    return 0;
}