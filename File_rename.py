import os
import json

def rename_images_and_remove_json(folder_path):
    image_count = 1
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            # JSON 파일이라면 삭제
            os.remove(os.path.join(folder_path, filename))
        elif filename.endswith('.jpg') or filename.endswith('.png'):
            # 이미지 파일이라면 순서대로 이름 변경
            # 새로운 파일 이름 생성
            new_filename = str(image_count) + os.path.splitext(filename)[1]
            # 원하는 한국말 추가
            new_filename_with_korean = f"횡단보도_{new_filename}"
            # 파일 이름 변경
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename_with_korean))
            image_count += 1

# 폴더 경로 입력
folder_path = 'C:\\Users\AIA\Desktop\교차로 데이터\(2차_최종) 교차로정보 데이터셋_20210720\횡단보도 이미지'

# 함수 호출
rename_images_and_remove_json(folder_path)