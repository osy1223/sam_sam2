import os
import torch
from hydra import initialize_config_dir, compose
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra

from matplotlib import font_manager, rc
# 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"  # macOS 한글 폰트 경로
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)


# Hydra 초기화
if GlobalHydra().is_initialized():
    GlobalHydra().clear()

# 절대 경로 설정
checkpoint_path = os.path.abspath("./checkpoints/sam2.1_hiera_base_plus.pt")
config_dir = os.path.abspath("./sam2/configs")
config_name = "sam2.1/sam2.1_hiera_b+.yaml"

# Hydra 초기화 및 구성 파일 로드
with initialize_config_dir(config_dir=config_dir, job_name="test", version_base="1.1"):
    cfg = compose(config_name=config_name)
    print("Configuration loaded successfully.")

# 모델 초기화
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_sam2(cfg=cfg, ckpt_path=checkpoint_path, device=device, mode="eval")
predictor = SAM2ImagePredictor(model)
print(f"Model loaded on device: {device}")

# 이미지 로드
image_path = os.path.abspath("./road.jpg")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# 이미지 표시 및 포인트 선택
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title("포인트를 선택하세요. 완료되면 창을 닫으세요.")
plt.axis('off')

# 포인트 선택
print("이미지에서 포인트를 선택하세요. 완료되면 창을 닫으세요.")
points = plt.ginput(n=-1, timeout=0)
plt.close()

# 선택한 포인트 처리
input_points = np.array(points)
print(f"선택된 포인트: {input_points}")

# 각 포인트에 대한 라벨 입력 (1: 포그라운드, 0: 백그라운드)
input_labels = []
for idx, point in enumerate(input_points):
    while True:
        try:
            label = int(input(f"포인트 {idx+1}의 라벨을 입력하세요 (1: 포그라운드, 0: 백그라운드): "))
            if label in [0, 1]:
                input_labels.append(label)
                break
            else:
                print("라벨은 0 또는 1이어야 합니다.")
        except ValueError:
            print("숫자를 입력해주세요.")
input_labels = np.array(input_labels)

# 예측 수행
with torch.inference_mode():
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

# 결과 시각화
plt.figure(figsize=(10, 10))
plt.imshow(image)
if masks is not None and len(masks) > 0:
    # plt.imshow(masks[0], cmap="jet", alpha=0.5)
    plt.imshow(masks[0], cmap="gray" )

plt.title("SAM2를 사용한 이미지 예측 결과")
plt.axis("off")
plt.show()
