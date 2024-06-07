from ultralytics import YOLO
import cv2
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# YOLO 모델 로드
model = YOLO('C:/Users/AIA/yolov5/runs/detect/train15/weights/best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 비디오 파일 로드
cap = cv2.VideoCapture('111111111.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5, vid_stride=7)

    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cls_id = int(box.cls[i])

            # 디버깅 출력
            print(f"Detected class ID: {cls_id}")

            # 클래스 ID가 model.model.names의 범위를 벗어나지 않도록 확인
            if cls_id < len(model.model.names):
                label = model.model.names[cls_id]
            else:
                label = "Unknown"

            conf = box.conf[i].item()

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}:{conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
