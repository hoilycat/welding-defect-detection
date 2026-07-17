#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── 유틸 ─────────────────────────────────────────────────────────────────────

fs::path pathFromUtf8(const std::string& utf8) {
#ifdef _WIN32
    if (utf8.empty()) return fs::path{};
    int size = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (size <= 0) return fs::path{};
    std::wstring wide(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], size);
    wide.pop_back();
    return fs::path(wide);
#else
    return fs::path(utf8);
#endif
}

json loadConfig() {
    std::ifstream f("config.json");
    if (!f.is_open()) {
        std::cerr << "[오류] config.json 없음! config.json.example 참고해서 만들어줘." << std::endl;
        exit(1);
    }
    json cfg; f >> cfg;
    return cfg;
}

// ── 라벨 인코딩 ───────────────────────────────────────────────────────────────

int labelEncode(const std::string& s) {
    if (s == "crack")    return 0;
    if (s == "porosity") return 1;
    if (s == "lack of fusion") return 2;
    if (s == "slag inclusion") return 3;
    return -1;
}

const std::vector<std::string> CLASS_NAMES = {"crack", "porosity", "fusion", "slag"};

cv::Scalar caseColor(const std::string& s) {
    if (s == "crack")    return cv::Scalar(0, 0, 255);   // 빨강
    if (s == "porosity") return cv::Scalar(0, 255, 0);   // 초록
    if (s == "lack of fusion" || s == "fusion") return cv::Scalar(255, 0, 0);   // 파랑
    if (s == "slag inclusion" || s == "slag")   return cv::Scalar(0, 255, 255); // 노랑
    return cv::Scalar(255, 255, 255);
}

// ── 특징 추출 ─────────────────────────────────────────────────────────────────
// 반환: [원형도, 종횡비, 밝기평균, 밝기표준편차, 정규화면적]  (5차원)

std::vector<float> extractFeatures(const cv::Mat& gray, const std::vector<cv::Point>& pts) {
    if (pts.size() < 3) return {};

    // 폴리곤 마스크 생성
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    std::vector<std::vector<cv::Point>> c = {pts};
    cv::fillPoly(mask, c, 255);

    // 원형도 = 4π·면적 / 둘레²  (원 → 1.0, 길쭉 → 0에 가까움)
    double area      = cv::contourArea(pts);
    double perimeter = cv::arcLength(pts, true);
    double circ      = (perimeter > 0) ? 4.0 * CV_PI * area / (perimeter * perimeter) : 0.0;

    // 종횡비 = 너비 / 높이 (바운딩 박스 기준)
    cv::Rect bbox = cv::boundingRect(pts);
    double ar = (bbox.height > 0) ? (double)bbox.width / bbox.height : 1.0;

    // 정규화 면적 (결함 크기 / 전체 이미지)
    double img_area  = (double)gray.rows * gray.cols;
    double norm_area = (img_area > 0) ? area / img_area : 0.0;

    // 마스크 영역 밝기 통계
    cv::Scalar mean_v, std_v;
    cv::meanStdDev(gray, mean_v, std_v, mask);

    return {
        (float)circ,
        (float)ar,
        (float)mean_v[0],
        (float)std_v[0],
        (float)norm_area
    };
}

// ── 시각화 모드 ───────────────────────────────────────────────────────────────

void processImage(const fs::path& img_path, const fs::path& json_path) {
    std::ifstream img_file(img_path, std::ios::binary);
    std::vector<char> buf(std::istreambuf_iterator<char>(img_file), {});
    cv::Mat img = cv::imdecode(cv::Mat(buf), cv::IMREAD_COLOR);
    if (img.empty()) { std::cout << "로드 실패: " << img_path.u8string() << std::endl; return; }

    std::ifstream jf(json_path);
    if (!jf.is_open()) { std::cout << "JSON 없음: " << json_path.u8string() << std::endl; return; }
    json j; jf >> j;

    cv::Mat gray, clahe_out, blurred, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(gray, clahe_out);
    cv::GaussianBlur(clahe_out, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 10, 40);

    cv::Mat overlay = img.clone();
    int cnt = 0;
    for (auto& ann : j["annotations"]) {
        auto& xs = ann["coordinate"]["x"];
        auto& ys = ann["coordinate"]["y"];
        std::string case_name = ann["case"];
        std::vector<cv::Point> pts;
        for (size_t i = 0; i < xs.size(); i++)
            pts.push_back({xs[i].get<int>(), ys[i].get<int>()});
        cv::polylines(overlay, pts, true, caseColor(case_name), 2);
        cv::putText(overlay, case_name, pts[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, caseColor(case_name), 1);
        cnt++;
    }

    cv::Mat gray_bgr, clahe_bgr, edges_bgr;
    cv::cvtColor(gray,      gray_bgr,  cv::COLOR_GRAY2BGR);
    cv::cvtColor(clahe_out, clahe_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(edges,     edges_bgr, cv::COLOR_GRAY2BGR);

    auto addLabel = [](cv::Mat& m, const std::string& t) {
        cv::putText(m, t, {10,25}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,255,255}, 2);
    };
    addLabel(img, "Original"); addLabel(gray_bgr, "Grayscale");
    addLabel(clahe_bgr, "CLAHE"); addLabel(edges_bgr, "Canny Edge"); addLabel(overlay, "GT Polygon");

    cv::Mat top, bottom, result;
    cv::hconcat(std::vector<cv::Mat>{img, gray_bgr, clahe_bgr}, top);
    cv::Mat blank = cv::Mat::zeros(img.size(), img.type());
    cv::hconcat(std::vector<cv::Mat>{edges_bgr, overlay, blank}, bottom);
    cv::vconcat(top, bottom, result);
    cv::resize(result, result, {}, 0.5, 0.5);

    std::cout << "파일: " << img_path.u8string() << " | 결함: " << cnt << std::endl;
    cv::imshow("Welding Defect Viewer", result);
    cv::waitKey(0);
}

// ── 데이터 수집 ───────────────────────────────────────────────────────────────

int collectFromFolder(const fs::path& img_dir, const fs::path& json_dir,
                      std::vector<std::vector<float>>& all_feats,
                      std::vector<int>& all_labels) {
    if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) {
        std::cout << "  [경고] 폴더 없음: " << img_dir.u8string() << std::endl;
        return 0;
    }
    int count = 0;
    for (auto& entry : fs::directory_iterator(img_dir)) {
        if (entry.path().extension() != ".jpg") continue;

        fs::path json_filename = entry.path().filename();
        json_filename.replace_extension(".json");
        fs::path json_path = json_dir / json_filename;

        // 이미지 로드 (한글 경로 대응)
        std::ifstream img_file(entry.path(), std::ios::binary);
        std::vector<char> buf(std::istreambuf_iterator<char>(img_file), {});
        cv::Mat gray = cv::imdecode(cv::Mat(buf), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;

        // CLAHE 전처리
        cv::Mat clahe_out;
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(gray, clahe_out);

        // JSON 파싱
        std::ifstream jf(json_path);
        if (!jf.is_open()) continue;
        json j; jf >> j;

        for (auto& ann : j["annotations"]) {
            std::string case_name = ann["case"];
            int label = labelEncode(case_name);
            if (label < 0) continue;

            auto& xs = ann["coordinate"]["x"];
            auto& ys = ann["coordinate"]["y"];
            std::vector<cv::Point> pts;
            for (size_t i = 0; i < xs.size(); i++)
                pts.push_back({xs[i].get<int>(), ys[i].get<int>()});

            auto feat = extractFeatures(clahe_out, pts);
            if (feat.empty()) continue;
            all_feats.push_back(feat);
            all_labels.push_back(label);
            count++;
        }
    }
    return count;
}

// ── SVM 학습 + 평가 ───────────────────────────────────────────────────────────

void trainAndEvaluate(const std::string& data_dir, const std::string& label_dir) {
    const int NUM_CLASSES = 4;

    struct FolderPair { fs::path img; fs::path lbl; };
    std::vector<FolderPair> folders = {
        {pathFromUtf8(u8"TS_RTAL_결함_1. 균열"),      pathFromUtf8(u8"TL_RTAL_결함_1. 균열")},
        {pathFromUtf8(u8"TS_RTAL_결함_2. 기공"),      pathFromUtf8(u8"TL_RTAL_결함_2. 기공")},
        {pathFromUtf8(u8"TS_RTAL_결함_3. 융합불량"),   pathFromUtf8(u8"TL_RTAL_결함_3. 융합불량")},
        {pathFromUtf8(u8"TS_RTAL_결함_4. 슬래그혼입"), pathFromUtf8(u8"TL_RTAL_결함_4. 슬래그혼입")},
    };

    std::vector<std::vector<float>> all_feats;
    std::vector<int> all_labels;

    fs::path base_img = pathFromUtf8(data_dir);
    fs::path base_lbl = pathFromUtf8(label_dir);

    std::cout << "\n[데이터 수집 중...]\n";
    for (auto& fp : folders) {
        int n = collectFromFolder(base_img / fp.img, base_lbl / fp.lbl, all_feats, all_labels);
        std::cout << "  " << fp.img.u8string() << " → " << n << "개 샘플" << std::endl;
    }

    if (all_feats.empty()) {
        std::cout << "\n샘플이 없습니다. 경로를 확인해주세요." << std::endl;
        return;
    }

    int total    = (int)all_feats.size();
    int feat_dim = (int)all_feats[0].size();
    std::cout << "\n총 샘플: " << total << "개 / 특징 차원: " << feat_dim << "\n";

    // 클래스별 샘플 수 출력
    std::vector<int> class_cnt(NUM_CLASSES, 0);
    for (int l : all_labels) if (l >= 0 && l < NUM_CLASSES) class_cnt[l]++;
    for (int i = 0; i < NUM_CLASSES; i++)
        std::cout << "  " << CLASS_NAMES[i] << ": " << class_cnt[i] << "개\n";

    // 셔플
    std::vector<int> idx(total);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(idx.begin(), idx.end(), rng);

    // cv::Mat 변환
    cv::Mat feat_mat(total, feat_dim, CV_32F);
    cv::Mat label_mat(total, 1, CV_32S);
    for (int i = 0; i < total; i++) {
        for (int j = 0; j < feat_dim; j++)
            feat_mat.at<float>(i, j) = all_feats[idx[i]][j];
        label_mat.at<int>(i, 0) = all_labels[idx[i]];
    }

    // Min-Max 정규화 (컬럼별)
    cv::Mat feat_norm = feat_mat.clone();
    for (int col = 0; col < feat_dim; col++) {
        double mn, mx;
        cv::minMaxLoc(feat_mat.col(col), &mn, &mx);
        if (mx - mn > 1e-8)
            feat_norm.col(col) = (feat_mat.col(col) - mn) / (mx - mn);
    }

    // Train / Test 분리 (80 : 20)
    int train_n = (int)(total * 0.8);
    int test_n  = total - train_n;
    cv::Mat train_feat = feat_norm.rowRange(0, train_n);
    cv::Mat test_feat  = feat_norm.rowRange(train_n, total);
    cv::Mat train_lbl  = label_mat.rowRange(0, train_n);
    cv::Mat test_lbl   = label_mat.rowRange(train_n, total);
    std::cout << "\n학습: " << train_n << "개 / 테스트: " << test_n << "개\n";

    // SVM 학습 (trainAuto: C, gamma 자동 탐색)
    std::cout << "\nSVM 학습 중 (RBF 커널, trainAuto)... 잠깐 걸릴 수 있어!\n";
    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setTermCriteria(cv::TermCriteria(
        cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 1000, 1e-6));
    auto train_data = cv::ml::TrainData::create(train_feat, cv::ml::ROW_SAMPLE, train_lbl);
    svm->trainAuto(train_data);
    std::cout << "학습 완료!\n";

    // 예측 & 정확도
    cv::Mat preds;
    svm->predict(test_feat, preds);

    int correct = 0;
    std::vector<std::vector<int>> conf(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
    for (int i = 0; i < test_n; i++) {
        int pred = (int)preds.at<float>(i);
        int gt   = test_lbl.at<int>(i, 0);
        if (pred == gt) correct++;
        if (gt >= 0 && gt < NUM_CLASSES && pred >= 0 && pred < NUM_CLASSES)
            conf[gt][pred]++;
    }
    double accuracy = (double)correct / test_n * 100.0;

    // 결과 출력
    std::cout << "\n========================================\n";
    std::cout << "  정확도: " << std::fixed << std::setprecision(1) << accuracy << "%\n";
    std::cout << "========================================\n\n";

    // Confusion Matrix
    std::cout << "Confusion Matrix (행=실제, 열=예측):\n";
    std::cout << std::setw(12) << "";
    for (auto& n : CLASS_NAMES) std::cout << std::setw(11) << n;
    std::cout << "\n";
    for (int r = 0; r < NUM_CLASSES; r++) {
        std::cout << std::setw(12) << CLASS_NAMES[r];
        for (int c = 0; c < NUM_CLASSES; c++)
            std::cout << std::setw(11) << conf[r][c];
        std::cout << "\n";
    }

    // 클래스별 정밀도 / 재현율
    std::cout << "\n클래스별 정밀도/재현율:\n";
    for (int i = 0; i < NUM_CLASSES; i++) {
        int tp = conf[i][i], fp = 0, fn = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (j != i) { fp += conf[j][i]; fn += conf[i][j]; }
        }
        double precision = (tp + fp > 0) ? (double)tp / (tp + fp) * 100 : 0;
        double recall    = (tp + fn > 0) ? (double)tp / (tp + fn) * 100 : 0;
        std::cout << "  " << std::setw(10) << CLASS_NAMES[i]
                  << " | 정밀도: " << std::setw(5) << std::setprecision(1) << precision << "%"
                  << " | 재현율: " << std::setw(5) << recall << "%\n";
    }

    // 모델 저장
    svm->save("svm_model.xml");
    std::cout << "\n모델 저장 완료: svm_model.xml\n";

    // result.json 저장 (Gradio 연결 준비)
    json results = json::array();
    for (int i = 0; i < test_n; i++) {
        int pred = (int)preds.at<float>(i);
        int gt   = test_lbl.at<int>(i, 0);
        cv::Rect bbox;  // 실제 bbox는 추후 이미지별 처리 시 채워짐

        json entry;
        entry["defect_type"] = (pred >= 0 && pred < NUM_CLASSES) ? CLASS_NAMES[pred] : "unknown";
        entry["gt_label"]    = (gt   >= 0 && gt   < NUM_CLASSES) ? CLASS_NAMES[gt]   : "unknown";
        entry["correct"]     = (pred == gt);
        entry["features"] = {
            {"circularity",   test_feat.at<float>(i, 0)},
            {"aspect_ratio",  test_feat.at<float>(i, 1)},
            {"mean_bright",   test_feat.at<float>(i, 2)},
            {"std_bright",    test_feat.at<float>(i, 3)},
            {"norm_area",     test_feat.at<float>(i, 4)}
        };
        results.push_back(entry);
    }

    json output;
    output["accuracy"]  = accuracy;
    output["total"]     = test_n;
    output["correct"]   = correct;
    output["stage"]     = "cpp_classical_vision";
    output["results"]   = results;

    std::ofstream out_file("result.json");
    out_file << output.dump(2);
    std::cout << "결과 저장 완료: result.json\n";
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    auto cfg = loadConfig();
    std::string data_dir  = cfg["data_dir"];
    std::string label_dir = cfg["label_dir"];

    std::cout << "==============================\n";
    std::cout << "  WeldVision  모드 선택\n";
    std::cout << "==============================\n";
    std::cout << "  1. 시각화 (이미지 뷰어)\n";
    std::cout << "  2. SVM 학습 + 평가\n";
    std::cout << "> " << std::flush;
    int mode; std::cin >> mode;

    if (mode == 1) {
        fs::path img_dir  = pathFromUtf8(data_dir)  / fs::path(u8"TS_RTAL_결함_1. 균열");
        fs::path json_dir = pathFromUtf8(label_dir) / fs::path(u8"TL_RTAL_결함_1. 균열");
        if (!fs::exists(img_dir) || !fs::is_directory(img_dir)) {
            std::cerr << "이미지 폴더가 없습니다: " << img_dir.u8string() << std::endl;
            return 1;
        }
        for (auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg") {
                fs::path json_filename = entry.path().filename();
                json_filename.replace_extension(".json");
                fs::path json_path = json_dir / json_filename;
                processImage(entry.path(), json_path);
            }
        }
    } else if (mode == 2) {
        trainAndEvaluate(data_dir, label_dir);
    } else {
        std::cout << "잘못된 입력!" << std::endl;
    }

    return 0;
}
