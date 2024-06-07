from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pyttsx3

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    


# YOLO 모델 로드
model = YOLO('C:/Users/AIA/yolov5/runs/detect/train15/weights/best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 실제 폰트 파일 경로로 변경하세요
font = ImageFont.truetype(font_path, 20)

# 비디오 파일 로드
cap = cv2.VideoCapture('111111111.mp4')

# LLM 초기화 (OpenAI API 키 필요)
llm = OpenAI(openai_api_key='your_openai_api_key')  # 여기에 자신의 OpenAI API 키를 입력하세요

# 프롬프트 템플릿 설정 (한국어)
prompt_template = PromptTemplate(
    input_variables=["objects"],
    template="탐지된 객체 목록이 주어졌습니다: {objects}. 중요도에 따라 객체를 우선순위대로 정렬하세요."
)

# LLM 체인 생성
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5, vid_stride=7)

    # 탐지된 객체를 저장할 리스트
    detected_objects = []

    # 탐지된 객체들을 순회하면서 리스트에 저장
    for det in results:
        box = det.boxes
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cls_id = int(box.cls[i])
            conf = box.conf[i].item()

            # 클래스 ID가 model.model.names의 범위를 벗어나지 않도록 확인
            if cls_id < len(model.model.names):
                label = model.model.names[cls_id]
            else:
                label = "Unknown"

            detected_objects.append((x1, y1, x2, y2, cls_id, conf, label))

    # 탐지된 객체 리스트를 문자열로 변환
    objects_str = ", ".join([obj[6] for obj in detected_objects])

    # 랭체인을 사용하여 객체 우선순위 결정
    response = llm_chain.run(objects=objects_str)
    sorted_labels = response.split(", ")

    # 객체를 우선순위에 따라 정렬
    detected_objects.sort(key=lambda x: (sorted_labels.index(x[6]) if x[6] in sorted_labels else len(sorted_labels), x[6]))

    # 프레임을 PIL 이미지로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 정렬된 순서대로 바운딩 박스와 라벨 그리기 및 사운드 재생
    for obj in detected_objects:
        x1, y1, x2, y2, cls_id, conf, label = obj

        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, y1 - 10), f"{label}:{conf:.2f}", font=font, fill=(0, 255, 0))

        # 사운드 재생 (첫 번째 우선순위 객체만)
        # if label == sorted_labels[0]:
        #     playsound.playsound('sound_file.mp3')  # 실제 사운드 파일 경로로 변경하세요

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
        # print(f"{location}, {label}")
        
        if x_center2 is not None and y_center2 is not None:
            speak(f"{location}, {label}")
            print(f"{location}, {label}")

    # PIL 이미지를 다시 OpenCV 이미지로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
