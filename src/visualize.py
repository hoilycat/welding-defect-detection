import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 경로 설정
DATA_ROOT = Path("D:/용접 데이터/127.창원 지역 특화산업 고도화 및 디지털 전환 촉진을 위한 용접 AI 학습 데이터/3.개방데이터/1.데이터")

TRAIN_IMG = DATA_ROOT / "Training/01.원천데이터"
TRAIN_LBL = DATA_ROOT / "Training/02.라벨링데이터"
VAL_IMG   = DATA_ROOT / "Validation/01.원천데이터"
VAL_LBL   = DATA_ROOT / "Validation/02.라벨링데이터"

# 결함 종류별 색상 (BGR)
COLORS = {
    "normal":               (0, 255, 0),    # 초록
    "crack":                (0, 0, 255),    # 빨강
    "porosity":             (255, 0, 0),    # 파랑
    "lack of fusion":       (0, 255, 255),  # 노랑
    "incomplete penetration": (255, 0, 255),# 보라
    "slag inclusion":       (255, 165, 0),  # 주황
    "undercut":             (0, 128, 255),  # 하늘
}

def visualize_sample(img_path: Path, json_path: Path):
    # 이미지 읽기 (한글 경로 우회)
    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"이미지 못 읽음: {img_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    info = label["image_data"]["information"]
    print(f"파일: {img_path.name} | 상태: {info}")

    # 폴리곤 그리기
    for ann in label["annotations"]:
        case = ann.get("case", "normal") or "normal"
        color = COLORS.get(case, (255, 255, 255))
        xs = ann["coordinate"]["x"]
        ys = ann["coordinate"]["y"]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        # 결함 이름 텍스트
        if case and case != "normal":
            cv2.putText(img, case, (xs[0], ys[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # BGR → RGB 변환 후 출력
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 6))
    plt.imshow(img_rgb)
    plt.title(f"{img_path.name} | {info}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # VTST 언더컷 폴더에서 첫 번째 파일 자동으로 찾기
    folder = TRAIN_IMG / "TS_VTST_결함_4. 언더컷"  
    files = list(folder.glob("*.jpg"))
    if files:
        test_img = files[0]
        test_lbl = TRAIN_LBL / "TL_VTST_결함_4. 언더컷" / (files[0].stem + ".json")
        visualize_sample(test_img, test_lbl)
    else:
        print("파일을 못 찾았어! 경로 확인해봐")