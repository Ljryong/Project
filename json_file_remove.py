import os
import glob


def delete_json_files(folder_path):
    # 해당 폴더 내의 모든 .json 파일 목록 가져오기
    json_files = glob.glob(os.path.join(folder_path, "*.json"))

    # 각 .json 파일 삭제
    for json_file in json_files:
        try:
            os.remove(json_file)
            print(f"Deleted: {json_file}")
        except OSError as e:
            print(f"Error: {json_file} : {e.strerror}")


# 사용 예시
folder_path = 'C:\\Users\AIA\Desktop\교차로 데이터\(2차_최종) 교차로정보 데이터셋_20210720\교차로정보 데이터셋_bbox_1'  # 원하는 폴더 경로로 바꿔주세요
delete_json_files(folder_path)
