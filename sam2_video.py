import os
import cv2
import torch
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt

# Hydra 초기화 해제 (중복 초기화 방지)
GlobalHydra.instance().clear()

# 모델 설정
checkpoint = os.path.abspath("./checkpoints/sam2.1_hiera_base_plus.pt")
config_dir = os.path.abspath("./sam2/configs")  # 절대 경로
config_name = "sam2.1/sam2.1_hiera_b+.yaml"  # 상대 경로

# Hydra 초기화 및 구성 파일 로드
with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
    cfg = compose(config_name=config_name)
    print("Configuration loaded successfully.")
    model = build_sam2(cfg, ckpt_path=checkpoint, device="mps", mode="eval")
    predictor = SAM2ImagePredictor(model)

# GPU/MPS 설정
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
predictor.model.to(device)
print(f"Model loaded on device: {device}")

# 비디오 처리
video_path = "demo/data/gallery/01_dog.mp4"  # 비디오 파일 경로
output_video_path = "01_dog_output_video.mp4"  # 결과 비디오 저장 경로

cap = cv2.VideoCapture(video_path)

# 첫 번째 프레임 가져오기
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the video file.")

# 사용자로부터 관심 지점 입력받기
plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
plt.title("Click on the points of interest and close the window when done.")
points = plt.ginput(n=-1, timeout=0, show_clicks=True)
plt.close()

# 좌표와 라벨 설정
input_points = np.array(points, dtype=np.int32)  # 클릭한 좌표를 배열로 변환
input_labels = np.ones(len(input_points), dtype=np.int32)  # 모든 좌표를 포그라운드로 설정
print(f"Selected points: {input_points}")

# 비디오 속성 설정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # OpenCV는 BGR로 이미지를 읽으므로 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_frame)

    with torch.inference_mode():
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

    # 마스크를 원본 프레임에 합성
    if masks is not None and len(masks) > 0:
        mask = masks[0].astype(np.uint8) * 255  # 첫 번째 마스크만 사용
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        combined_frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)
    else:
        combined_frame = frame

    # 비디오에 프레임 저장
    out.write(combined_frame)

# 리소스 정리
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing complete. Saved to {output_video_path}.")
