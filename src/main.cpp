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
    // 폴리라인 그리기
    cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), 2);
}
    // 결과 이미지 표시
    cv::imshow("result", img);

    cv::Mat gray, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 50, 150);
    
    cv::imshow("gray", gray);
    cv::imshow("blurred", blurred);
    cv::imshow("edges", edges);
    
    // 컨투어 찾기
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& contour : contours){
        cv::Rect box = cv::boundingRect(contour);
        cv::rectangle(img, box, cv::Scalar(255, 0, 0), 2);
    }

    //특징 추출
    for (auto& contour : contours){
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        cv::Rect box = cv::boundingRect(contour);
        double aspect_ratio = (double)box.width / box.height;

        std::cout << "면적: " << area
                  << "|둘레: " << perimeter
                  << "|가로세로비: " << aspect_ratio 
                  << std::endl;

        // 라벨링 정보 출력
        std::string label;
        if(area < 500){
            label = "Small Defect";
        } else if(aspect_ratio > 2.0 || aspect_ratio <0.5){
            label = "Crack";
        } else {
            label = "Porosity";
        }
    

    std::cout << "결함 유형: " << label << std::endl;

    cv::Rect box2 = cv::boundingRect(contour);
    cv::putText(img, label, cv::Point(box2.x, box2.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

    cv::imshow("result2", img);

    cv::waitKey(0);
    return 0;
    }  

}