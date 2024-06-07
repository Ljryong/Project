from flask import Flask, render_template, Response
import cv2
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# 웹캠 비디오 스트리밍 함수
def video_stream():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV의 프레임을 JPEG 이미지로 변환
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # JPEG 이미지를 base64로 인코딩하여 HTML에 전달
        img_base64 = base64.b64encode(frame_bytes)
        img_base64 = img_base64.decode('utf-8')

        # HTML에 렌더링할 이미지 데이터 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
