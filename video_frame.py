import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    # 비디오 파일 열기
    vidcap = cv2.VideoCapture(video_path)
    
    if not vidcap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    success, image = vidcap.read()
    count = 0
    saved_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while success:
        # 지정된 프레임 간격에 따라 이미지를 저장
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame{saved_count:04d}.jpg")
            if image is None:
                print(f"Error: Empty image frame at frame {count}")
                continue
            success_save = cv2.imwrite(frame_filename, image)
            if success_save:
                print(f"Successfully saved frame {saved_count} at {frame_filename}")
            else:
                print(f"Failed to save frame {saved_count} at {frame_filename}")
            saved_count += 1
        success, image = vidcap.read()
        count += 1

    vidcap.release()
    print(f"Frame extraction complete. Total frames saved: {saved_count}")

    # 저장된 프레임의 경로와 마지막 몇 개의 프레임을 출력
    last_frames = os.listdir(output_folder)[-10:]  # 마지막 10개 프레임을 출력
    print(f"Frames saved in: {os.path.abspath(output_folder)}")
    print("Last few frames saved:")
    for frame in last_frames:
        print(frame)

# 사용 예시
video_path = r'C:/Project/횡단보도_신호등2.mp4'  # 동영상 파일 경로
output_folder = r'C:/Users/AIA/Desktop/교차로 데이터/횡단보도_신호등'  # 프레임이 저장될 폴더 (raw 문자열 사용)
frame_interval = 10  # 몇 프레임 단위로 저장할지 설정 (예: 10프레임마다 저장)

extract_frames(video_path, output_folder, frame_interval)
