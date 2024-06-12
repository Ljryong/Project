from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# YOLO 모델 로드
model = YOLO('C:/Users/AIA/yolov5/runs/detect/train16/weights/best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 클래스별 바운딩 박스 색상 지정
colors = {
    '빨간불': (255, 0, 0),  # 빨간색
    '초록불': (0, 255, 0),  # 초록색
    '자전거': (0, 0, 0),  # 검정색
    '킥보드': (128, 0, 128),  # 보라색
    '라바콘': (255, 165, 0),  # 주황색
    '횡단보도': (255, 255, 255)  # 흰색
}


# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 여기서 "path/to/your/font.ttf"를 실제 한글 폰트 파일 경로로 변경하세요.
font = ImageFont.truetype(font_path, 20)

# 비디오 파일 로드
cap = cv2.VideoCapture('111111111.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5, vid_stride=7)

    # 프레임을 PIL 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cls_id = int(box.cls[i])

            # 클래스 ID가 model.model.names의 범위를 벗어나지 않도록 확인
            if cls_id < len(model.model.names):
                label = model.model.names[cls_id]
                color = colors.get(label, (128, 128, 128))  # 기본 색상은 회색
            else:
                label = "Unknown"
                color = (128, 128, 128)  # 기본 색상은 회색

            conf = box.conf[i].item()
            
            # 텍스트 크기 계산
            bbox = draw.textbbox((0, 0), f"{label}:{conf:.2f}", font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - text_height - 10), f"{label}:{conf:.2f}", font=font, fill=color)

            # 중심 좌표 계산 및 위치 표시
            x_center1 = (x1 + x2) / 2
            y_center1 = (y1 + y2) / 2
            if x_center1 < 720 / 2:
                x_center2 = '좌'
            elif 720 / 2 < x_center1 < 720:
                x_center2 = '정면'
            else:
                x_center2 = '우'
            if y_center1 < 720 / 2:
                y_center2 = '상'
            elif 720 / 2 < y_center1 < 720:
                y_center2 = '중'
            else:
                y_center2 = '하'
            location = f"{x_center2} {y_center2}"
            print(f"{location}, {label}")

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()