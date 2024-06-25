from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# OpenAI API 키 설정
openai_api_key =  'api_key'

# LangChain 인터페이스 정의
def get_classes_from_user(prompt: str) -> list:
    template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0.5)
    chain = LLMChain(llm=llm, prompt=template)

    response = chain.run(prompt)
    classes = response.split(", ")
    return [cls.strip() for cls in classes]

prompt = "탐지할 클래스를 쉼표로 구분하여 입력해주세요 (예: '횡단보도, 빨간불'): "
desired_classes = get_classes_from_user(prompt)
print("탐지할 클래스:", desired_classes)

import streamlit as st
import threading
import cv2
import torch
import pyttsx3
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import os

os.environ['OPENAI_API_KEY'] = openai_api_key

# YOLO 모델 로드
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# pyttsx3 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 250)

tts_active = threading.Event()

def speak_location_and_label(location, label):
    tts_active.set()  # TTS가 활성 상태임을 표시
    engine.say(f"{location}, {label}")
    engine.runAndWait()
    tts_active.clear()  # TTS가 비활성 상태임을 표시

# YOLO 모델 로드
model = YOLO('best.pt')
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 클래스별 바운딩 박스 색상 지정
colors = {
    '빨간불': (255, 0, 0),  # 빨간색
    '초록불': (0, 255, 0),  # 초록색
    '자전거': (0, 0, 0),  # 검정색
    '킥보드': (128, 0, 128),  # 보라색
    '라바콘': (255, 165, 0),  # 주황색
    '횡단보도': (255, 255, 255)  # 횡단보도는 흰색으로 설정
}

def detect_objects_in_video(video_path, desired_classes):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO를 사용하여 프레임에서 객체 탐지
        results = model.predict(frame, conf=0.5)

        detected_labels = set()
        label_location_map = {}  # label과 location을 저장하기 위한 딕셔너리

        for det in results:
            box = det.boxes
            for i in range(len(box.xyxy)):
                x1, y1, x2, y2 = box.xyxy[i].tolist()
                cls_id = int(box.cls[i])

                if cls_id < len(model.model.names):
                    label = model.model.names[cls_id]
                    color = colors.get(label, (128, 128, 128))  # 기본 색상은 회색
                else:
                    label = "Unknown"
                    color = (128, 128, 128)  # 기본 색상은 회색

                conf = box.conf[i].item()
                detected_labels.add(label)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

                x_center1 = (x1 + x2) / 2
                y_center1 = (y1 + y2) / 2
                if x_center1 < 640 / 3:
                    x_center2 = '좌'
                elif 640 / 3 < x_center1 < 640 / 3 + 640 / 3:
                    x_center2 = '정면'
                else:
                    x_center2 = '우'
                if y_center1 < 1200 / 3:
                    y_center2 = '상'
                elif 1200 / 3 < y_center1 < 1200 / 3 + 1200 / 3:
                    y_center2 = '중'
                else:
                    y_center2 = '하'
                location = f"{x_center2} {y_center2}"
                output_text = f"{location} {label}"
                print(output_text)

                label_location_map[label] = location  # label과 location을 딕셔너리에 저장

                fontpath = "/usr/share/fonts/X11/Type1/c0419bt_.pfb"
                font = ImageFont.truetype(fontpath, 20)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((int(x1), int(y1) - 25), output_text, font=font, fill=color)
                frame = np.array(img_pil)

        # 감지된 라벨에 대해 TTS 처리
        if '빨간불' in detected_labels and '횡단보도' in detected_labels:
            # 빨간불과 횡단보도가 동시에 감지된 경우, 빨간불만 처리
            if not tts_active.is_set():
                threading.Thread(target=speak_location_and_label, args=(label_location_map['빨간불'], '빨간불')).start()
        elif '초록불' in detected_labels and '횡단보도' in detected_labels:
            # 초록불만 감지된 경우
            if not tts_active.is_set():
                threading.Thread(target=speak_location_and_label, args=(label_location_map['초록불'], '초록불')).start()
        else:
            # 위의 모든 경우에 해당하지 않는 라벨 처리
            for label in detected_labels:
                if not tts_active.is_set():
                    threading.Thread(target=speak_location_and_label, args=(label_location_map[label], label)).start()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# LangChain 인터페이스 정의
def get_classes_from_user(prompt: str) -> list:
    template = PromptTemplate(input_variables=["prompt"], template="{prompt}")
    llm = OpenAI(temperature=0.5)  # API 키 설정 필요
    chain = LLMChain(llm=llm, prompt=template)

    response = chain.run(prompt)
    classes = response.split(", ")
    return [cls.strip() for cls in classes]

st.title("YOLO Object Detection App")

video_source = st.radio("비디오 소스 선택", ('캠코더', '비디오 파일'))

if video_source == '비디오 파일':
    video_file = st.file_uploader("비디오 파일 업로드", type=["mp4", "mov", "avi"])
    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())
        st.success("비디오 파일 업로드 완료")

prompt = "탐지할 클래스를 쉼표로 구분하여 입력해주세요 (예: '횡단보도, 빨간불'): "
desired_classes = get_classes_from_user(prompt)
classes = st.text_input(f"탐지할 클래스: {', '.join(desired_classes)}")
# st.text(f"탐지할 클래스: {', '.join(desired_classes)}")
if st.button("탐지 시작"):
    if video_source == '캠코더':
        detect_objects_in_video(0, desired_classes)  # 0은 웹캠을 의미
    else:
        detect_objects_in_video("temp_video.mp4", desired_classes)

# 음성 출력 스레드 종료
print("Processing complete.")
