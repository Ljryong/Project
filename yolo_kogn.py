import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("C:/Users/AIA/yolov5/runs/detect/train14/weights/best.pt" )

# 비디오 파일 열기
video_path = "111111111.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 저장 설정
output_path = "output_video1-1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 신뢰도 임계값 설정
confidence_threshold = 0.3

# 비디오 프레임 루프
while cap.isOpened():
    # 비디오에서 프레임 읽기
    success, frame = cap.read()

    if success:
        # YOLOv8 추론 수행
        results = model(frame)
        
        # 필터링된 결과 생성
        filtered_boxes = []
        if results and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                if box.conf > confidence_threshold:
                    filtered_boxes.append(box)
        
        # 필터링된 결과 시각화
        if filtered_boxes:
            annotated_frame = results[0].plot(boxes=filtered_boxes)
        else:
            annotated_frame = frame

        # 결과 프레임을 비디오에 저장
        out.write(annotated_frame)

        # 결과 프레임을 화면에 표시
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 비디오 끝에 도달하면 루프 종료
        break

# 비디오 캡처 객체와 비디오 라이터 객체 해제 및 창 닫기
cap.release()
out.release()
cv2.destroyAllWindows()