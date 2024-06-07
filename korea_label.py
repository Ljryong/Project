from ultralytics import YOLO
import cv2
""" from playsound import playsound  # 수정된 부분
from gtts import gTTS

# YOLO 모델 로드
def text_to_speech(text):
    if text:  # 텍스트가 비어있지 않을 경우에만 TTS 모듈 호출
        tts = gTTS(text=text, lang='ko')
        output_path = 'sound/xy_class4.mp3'
        tts.save(output_path)
        return output_path
    else:
        return None """

model = YOLO('C:/Users/AIA/yolov5/runs/detect/train15/weights/best.pt')

# 클래스 이름 설정
model.model.names = ['빨간불', '초록불', '빨간불', '초록불', '자전거', '킥보드', '라바콘', '횡단보도']

# 비디오 파일 로드
cap = cv2.VideoCapture('111111111.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5)

    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes

        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cv2.rectangle(det.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cls_id = int(box.cls[i])
            class_name = model.model.names[cls_id]
            label = f"{class_name}"
            # cv2.putText(det.orig_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
            output_text = f"{location} {label}"
            print(output_text)
            # if x_center2 is not None and y_center2 is not None:
            #     sound_path = text_to_speech(output_text)
            #     if sound_path:
            #         # playsound(sound_path)
            #         print(output_text)

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
