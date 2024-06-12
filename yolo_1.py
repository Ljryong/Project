import os
import queue
import threading
from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# YOLO 모델 로드
model_path = r'C:\project\yolov8\runs\detect\yolo8s_img640_batch32\weights\best.pt'
model = YOLO(model_path)
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 클래스별 바운딩 박스 색상 지정
colors = {
    '빨간불': (255, 0, 0),
    '초록불': (0, 255, 0),
    '자전거': (0, 0, 0),
    '킥보드': (128, 0, 128),
    '라바콘': (255, 165, 0),
    '횡단보도': (255, 255, 255)
}

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 20)

# pyttsx3 엔진 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 300)
engine.setProperty('volume', 1.0)

# 큐 초기화
q = queue.Queue()

def speak():
    while True:
        text = q.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# 음성 출력 스레드 시작
speak_thread = threading.Thread(target=speak)
speak_thread.start()

# 비디오 파일 로드
cap = cv2.VideoCapture(r"C:\Users\AIA\Desktop\test3.mp4")

# 상태 추적 변수
current_light_state = None  # 'red', 'green' 또는 None

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

    # 현재 프레임에서의 상태 변수
    is_crosswalk_detected = False
    is_red_light_detected = False
    is_green_light_detected = False

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

            # 상태 업데이트
            if label == '횡단보도' and conf >= 0.6:
                is_crosswalk_detected = True
            if label == '빨간불' and conf >= 0.6:
                is_red_light_detected = True
            if label == '초록불' and conf >= 0.6:
                is_green_light_detected = True

    # 음성 출력 조건 확인 및 큐에 추가
    if is_crosswalk_detected:
        if is_red_light_detected and current_light_state != 'red':
            # 큐 비우기
            while not q.empty():
                q.get()
            current_light_state = 'red'
            q.put("빨간불이니 기다려 주세요")
        elif is_green_light_detected and current_light_state != 'green':
            # 큐 비우기
            while not q.empty():
                q.get()
            current_light_state = 'green'
            q.put("초록불로 바뀌었으니 길을 건너세요")
        elif is_red_light_detected and current_light_state == 'red':
            q.put("빨간불이니 기다려 주세요")
        elif is_green_light_detected and current_light_state == 'green':
            q.put("초록불이니 길을 건너세요")

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 음성 출력 스레드 종료
q.put(None)
speak_thread.join()

print("Processing complete.")