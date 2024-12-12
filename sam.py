import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch

# MPS 또는 CPU 사용 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"

# SAM 모델 로드
# sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")

sam.to(device=device)
predictor = SamPredictor(sam)

# 이미지 로드
image_path = "road.jpg"  # 사용할 이미지 경로
image = cv2.imread(image_path)
print('Image size: ', image.shape)
predictor.set_image(image)

# 클릭된 좌표 저장 리스트
click_points = []
labels = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x,y])
        labels.append(1)
        print(f'Point added: ({x}, {y})')

# OpenCV 창 생성 및 마우스 콜백 등록
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

print("Click on the image to add points. Press 'q' to finish.")
while True:
    # 클릭된 좌표 표시
    temp_image = image.copy()
    for point in click_points:
        cv2.circle(temp_image, tuple(point), 5, (0, 0, 255), -1)  # 클릭된 좌표 표시
    cv2.imshow("Image", temp_image)

    # 'q' 키를 누르면 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.setMouseCallback('Image', lambda *args: None)
cv2.destroyAllWindows()

# 클릭된 좌표를 SAM에 전달
input_points = np.array(click_points)
input_labels = np.array(labels)

print("Predicting segmentation...")
# SAM으로 세그먼트 예측
masks, _, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False
)
print('Segmentation prediction complete')

print('Visualizing results...')
# 세그먼트 마스크 출력
plt.imshow(image)
plt.imshow(masks[0], cmap="gray") 
plt.show()
