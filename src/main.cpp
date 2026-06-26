#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("test.jpg");

    if (img.empty()) {
        std::cout << "이미지 로드 실패" << std::endl;
        return -1;
    }

    cv::imshow("test", img);
    cv::waitKey(0);
    return 0;
}