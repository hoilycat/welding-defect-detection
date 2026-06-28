#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void processImage(const std::string& img_path, const std::string& json_path) {
    // 이미지 읽기
    std::ifstream img_file(img_path, std::ios::binary);
    std::vector<char> buf(std::istreambuf_iterator<char>(img_file), {});
    cv::Mat img = cv::imdecode(cv::Mat(buf), cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "로드 실패: " << img_path << std::endl;
        return;
    }

    // JSON 읽기
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        std::cout << "JSON 없음: " << json_path << std::endl;
        return;
    }
    nlohmann::json j;
    json_file >> j;

    // 전처리
    cv::Mat gray, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 30, 100);

    // 컨투어
    std::vector<std::vector<cv::Point>> contours;

    // 디버깅용 이미지 저장
    cv::imwrite("/Users/yong-yong/프로젝트/welding-defect-detection/debug_edges.jpg", edges);
    cv::imwrite("/Users/yong-yong/프로젝트/welding-defect-detection/debug_gray.jpg", gray);

    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 출력 확인
    std::cout << "컨투어 수: " << contours.size() << std::endl;
    std::cout << "이미지 크기: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "컨투어 수: " << contours.size() << std::endl;

    for (auto& contour : contours) {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        cv::Rect box = cv::boundingRect(contour);
        double aspect_ratio = (double)box.width / box.height;


        std::string label;
        if (area < 500) {
            label = "Small Defect";
        } else if (aspect_ratio > 2.0 || aspect_ratio < 0.5) {
            label = "Crack";
        } else {
            label = "Porosity";
        }

        std::cout << "파일: " << img_path
                  << " | 유형: " << label
                  << " | 면적: " << area << std::endl;

        cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2);
        cv::putText(img, label, cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("result", img);
    cv::waitKey(1000);
}

int main() {
    std::string img_dir = "/Users/yong-yong/Documents/127.창원 지역 특화산업 고도화 및 디지털 전환 촉진을 위한 용접 AI 학습 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_RTAL_결함_1. 균열";
    std::string json_dir = "/Users/yong-yong/Documents/127.창원 지역 특화산업 고도화 및 디지털 전환 촉진을 위한 용접 AI 학습 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_RTAL_결함_1. 균열";

    for (auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg") {
            std::string img_path = entry.path().string();
            std::string filename = entry.path().stem().string();
            std::string json_path = json_dir + "/" + filename + ".json";
            processImage(img_path, json_path);
        }
    }

    return 0;

}