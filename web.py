import os
import queue
import threading
from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3
import openai
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from flask import Flask, render_template, Response

# OpenAI API 키 설정
openai.api_key = ''

# YOLO 모델 로드
model_path = r'C:\Users\AIA\yolov5\runs\detect\train16\weights\best.pt'
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
engine.setProperty('rate', 200)

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

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)  # 웹캠 사용

# 상태 추적 변수
current_light_state = None  # 'red', 'green' 또는 None

# 랭체인 LLM 설정
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai.api_key
)

prompt_template = PromptTemplate(
    input_variables=["command"],
    template="사용자가 다음 명령을 내렸습니다: {command}. 어떤 클래스만 탐지해야 하나요?"
)

llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

def process_command(command):
    try:
        response = llm_chain.run(command)
        response_text = response.strip()
        print(f"Response from OpenAI: {response_text}")
        
        # '횡단보도'와 '빨간불' 감지 요청에 대한 처리
        if "횡단보도" in response_text and "빨간불" in response_text:
            return ["횡단보도", "빨간불"]
        else:
            return []
    except Exception as e:
        print(f"Error processing command: {str(e)}")
        return []

desired_classes = process_command("횡단보도와 빨간불만 탐지해줘")
print("Classes to detect:", desired_classes)

def filter_detections_by_class(detections, classes):
    filtered_results = []
    for det in detections:
        boxes = det.boxes
        for i in range(len(boxes.xyxy)):
            cls_id = int(boxes.cls[i])
            label = model.model.names[cls_id]
            if label in classes:
                filtered_results.append({
                    'box': boxes.xyxy[i].tolist(),
                    'label': label,
                    'confidence': boxes.conf[i].item()
                })
    return filtered_results

def generate_frames():
    global current_light_state
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # YOLO를 사용하여 프레임에서 객체 탐지
        results = model.predict(frame, conf=0.5, vid_stride=7)
        filtered_results = filter_detections_by_class(results, desired_classes)

        # 프레임을 PIL 이미지로 변환
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 현재 프레임에서의 상태 변수
        is_crosswalk_detected = False
        is_red_light_detected = False
        is_green_light_detected = False

        # 필터링된 결과 순회
        for det in filtered_results:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']
            color = colors.get(label, (128, 128, 128))  # 기본 색상은 회색

            # 바운딩 박스 및 텍스트 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            text = f"{label}: {conf:.2f}"
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((x1, y1 - text_height), text, fill=color, font=font)

            # 상태 업데이트 및 음성 큐 추가
            if label == '횡단보도' and conf >= 0.6:
                is_crosswalk_detected = True
                q.put(f"횡단보도가 감지되었습니다")  # 감지된 횡단보도에 대한 텍스트 추가
            if label == '빨간불' and conf >= 0.6:
                is_red_light_detected = True

        # 음성 출력 조건 확인 및 큐에 추가
        if is_crosswalk_detected:
            if is_red_light_detected and current_light_state != 'red':
                while not q.empty():
                    q.get()
                current_light_state = 'red'
                q.put("빨간불이니 기다려 주세요")
            elif is_green_light_detected and current_light_state != 'green':
                while not q.empty():
                    q.get()
                current_light_state = 'green'
                q.put("초록불로 바뀌었으니 길을 건너세요")

        # PIL 이미지를 다시 OpenCV 이미지로 변환
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 프레임 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # 프레임 스트리밍
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask 앱 설정
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# 음성 출력 스레드 종료
q.put(None)
speak_thread.join()

print("Processing complete.")
