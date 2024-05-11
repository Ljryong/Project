from ultralytics import YOLO
import cv2

# YOLO 모델 로드
# model = YOLO('yolov9c.pt')
model = YOLO('runs/detect/train8/weights/best.pt')
model.model.names = ['빨간불','초록불','빨간불','초록불','자전거','킥보드','라바콘','횡단보도']
# 비디오 파일 로드
cap = cv2.VideoCapture('video/working.mp4')

# 프레임별로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO를 사용하여 프레임에서 객체 탐지
    results = model.predict(frame, conf=0.5)
    names = model.names  # 클래스 이름 목록

    # 탐지된 객체들을 순회하면서
    for det in results:
        box = det.boxes
        conf = box.conf
        for i in range(len(box.xyxy)):
            x1, y1, x2, y2 = box.xyxy[i].tolist()
            cls_id = int(box.cls[i])  # 클래스 번호
            cls_name = names[cls_id]  # 클래스 이름

            cv2.rectangle(det.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{cls_name}"
            cv2.putText(det.orig_img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x_center1 = (x1 + x2) / 2
            y_center1 = (y1 + y2) / 2

            if x_center1 < 720 / 2:
                x_center2 = 'left'
            elif 720 / 2 < x_center1 < 720 / 2 + 720 / 2:
                x_center2 = 'middle'
            elif 720 / 2 < x_center1:
                x_center2 = 'right'

            if y_center1 < 720 / 2:
                y_center2 = 'up'
            elif 720 / 2 < y_center1 < 720 / 2 + 720 / 2:
                y_center2 = 'center'
            elif 720 / 2 < y_center1:
                y_center2 = 'down'

            location = f"{x_center2} {y_center2}"
            print(f"{location}, {label}")

    # 프레임 출력
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
