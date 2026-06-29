#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <windows.h>

namespace fs = std::filesystem;

// UTF-8 문자열 → Windows 와이드 문자열 변환
std::wstring to_wide(const std::string& utf8) {
    if (utf8.empty()) return L"";
    int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    std::wstring wide(size - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], size);
    return wide;
}

// config.json에서 경로 읽기
nlohmann::json loadConfig() {
    std::ifstream f("config.json");
    if (!f.is_open()) {
        std::cerr << "[오류] config.json 없음! config.json.example 참고해서 만들어줘." << std::endl;
        exit(1);
    }
    nlohmann::json cfg;
    f >> cfg;
    return cfg;
}

// case 이름 → 색상
cv::Scalar caseColor(const std::string& case_name) {
    if (case_name == "crack")    return cv::Scalar(0, 0, 255);    // 빨강
    if (case_name == "porosity") return cv::Scalar(0, 255, 0);    // 초록
    if (case_name == "fusion")   return cv::Scalar(255, 0, 0);    // 파랑
    if (case_name == "slag")     return cv::Scalar(0, 255, 255);  // 노랑
    return cv::Scalar(255, 255, 255);                              // 흰색
}

void processImage(const std::string& img_path, const fs::path& json_path, const std::string& output_dir) {
    // 이미지 읽기 (한글 경로 대응)
    std::ifstream img_file(img_path, std::ios::binary);
    std::vector<char> buf(std::istreambuf_iterator<char>(img_file), {});
    cv::Mat img = cv::imdecode(cv::Mat(buf), cv::IMREAD_COLOR);

    if (img.empty()) {
        std::cout << "로드 실패: " << img_path << std::endl;
        return;
    }

    // JSON 읽기 (fs::path 그대로 사용 → 한글 경로 안전)
    std::ifstream json_file(json_path);
    if (!json_file.is_open()) {
        std::cout << "JSON 없음: " << json_path << std::endl;
        return;
    }
    nlohmann::json j;
    json_file >> j;

    // 전처리 뷰 생성
    cv::Mat gray, clahe_out, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(gray, clahe_out);

    // 블러 + Canny 엣지 검출
    cv::GaussianBlur(clahe_out, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 10, 40);

    // 폴리곤 그릴 오버레이 이미지 (원본 복사)
    cv::Mat overlay = img.clone();
    int defect_count = 0;
    for (auto& ann : j["annotations"]) {
        auto& xs = ann["coordinate"]["x"];
        auto& ys = ann["coordinate"]["y"];
        std::string case_name = ann["case"];

        std::vector<cv::Point> pts;
        for (size_t i = 0; i < xs.size(); i++)
            pts.push_back(cv::Point(xs[i].get<int>(), ys[i].get<int>()));

        cv::Scalar color = caseColor(case_name);
        cv::polylines(overlay, pts, true, color, 2);
        cv::putText(overlay, case_name, pts[0],
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        defect_count++;
    }

    // 흑백 → BGR 변환 (hconcat용)
    cv::Mat gray_bgr, clahe_bgr, edges_bgr;
    cv::cvtColor(gray,      gray_bgr,  cv::COLOR_GRAY2BGR);
    cv::cvtColor(clahe_out, clahe_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(edges,     edges_bgr, cv::COLOR_GRAY2BGR);

    // 라벨 추가
    auto addLabel = [](cv::Mat& m, const std::string& text) {
        cv::putText(m, text, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
    };
    addLabel(img,       "Original");
    addLabel(gray_bgr,  "Grayscale");
    addLabel(clahe_bgr, "CLAHE");
    addLabel(edges_bgr, "Canny Edge");
    addLabel(overlay,   "GT Polygon");

    // 상단: 원본 | 그레이 | CLAHE  /  하단: 엣지 | 폴리곤 (가운데 정렬)
    cv::Mat top, bottom, result;
    cv::hconcat(std::vector<cv::Mat>{img, gray_bgr, clahe_bgr}, top);

    // 하단은 2개라 빈칸 채우기
    cv::Mat blank = cv::Mat::zeros(img.size(), img.type());
    cv::hconcat(std::vector<cv::Mat>{edges_bgr, overlay, blank}, bottom);

    cv::vconcat(top, bottom, result);

    // 화면에 맞게 축소
    cv::resize(result, result, cv::Size(), 0.5, 0.5);

    std::cout << "파일: " << img_path << " | 결함 수: " << defect_count << std::endl;

    cv::imshow("Welding Defect Viewer", result);
    cv::waitKey(0);
}

int main() {
    auto cfg = loadConfig();
    std::string data_dir   = cfg["data_dir"];
    std::string label_dir  = cfg["label_dir"];
    std::string output_dir = cfg["output_dir"];

    SetConsoleOutputCP(CP_UTF8);

    std::wstring base_wide = to_wide(data_dir);
    fs::path img_dir  = fs::path(base_wide) / L"TS_RTAL_결함_1. 균열";
    fs::path json_dir = fs::path(to_wide(label_dir)) / L"TL_RTAL_결함_1. 균열";

    // Windows API로 직접 존재 확인
    DWORD attr = GetFileAttributesW(img_dir.wstring().c_str());
    bool exists = (attr != INVALID_FILE_ATTRIBUTES);

    std::wcout << L"이미지 폴더: " << img_dir.wstring() << std::endl;
    std::cout  << "폴더 존재: " << (exists ? "YES" : "NO") << std::endl;

    if (!exists) {
        // base 경로 확인
        DWORD base_attr = GetFileAttributesW(base_wide.c_str());
        std::cout << "base 경로 존재: " << (base_attr != INVALID_FILE_ATTRIBUTES ? "YES" : "NO") << std::endl;
        return 1;
    }

    for (auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() == ".jpg") {
            std::string img_path = entry.path().string();
            std::string filename = entry.path().stem().string();
            fs::path json_path   = json_dir / (filename + ".json");
            processImage(img_path, json_path, output_dir);
        }
    }

    return 0;
}