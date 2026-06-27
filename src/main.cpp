#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // 이미지 읽기 (한글 경로)
    std::string img_path = "/Users/yong-yong/Documents/127.창원 지역 특화산업 고도화 및 디지털 전환 촉진을 위한 용접 AI 학습 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_RTAL_결함_1. 균열/RT_AL_01_14487416.jpg";

    std::ifstream img_file(img_path, std::ios::binary);
    std::vector<char> buf(std::istreambuf_iterator<char>(img_file), {});
    cv::Mat img = cv::imdecode(cv::Mat(buf), cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "이미지 로드 실패" << std::endl;
        return -1;
    }

    // JSON 읽기
    std::string json_path = "/Users/yong-yong/Documents/127.창원 지역 특화산업 고도화 및 디지털 전환 촉진을 위한 용접 AI 학습 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_RTAL_결함_1. 균열/RT_AL_01_14487416.json";

    std::ifstream json_file(json_path);
    nlohmann::json j;
    json_file >> j;

    // 폴리곤 그리기
for (auto& annotation : j["annotations"]) {
    auto xs = annotation["coordinate"]["x"];
    auto ys = annotation["coordinate"]["y"];

    std::vector<cv::Point> pts;
    for (int i = 0; i < xs.size(); i++) {
        pts.push_back(cv::Point(xs[i], ys[i]));
    }
    cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), 2);
}

    cv::imshow("result", img);
    cv::waitKey(0);
    return 0;
}  