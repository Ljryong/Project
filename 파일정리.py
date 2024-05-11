import os
import glob


def delete_files_except_matching_txt(folder_path):
    # txt 파일들의 기본 이름을 추출하여 set에 저장
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    txt_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in txt_files)

    # 폴더 내 모든 파일 확인
    all_files = glob.glob(os.path.join(folder_path, '*'))

    for file_path in all_files:
        if not os.path.isfile(file_path):
            continue

        base_name, ext = os.path.splitext(os.path.basename(file_path))

        # txt 파일이거나 txt와 이름이 같은 파일인 경우 제외
        if ext == '.txt' or base_name in txt_basenames:
            continue

        # 나머지 파일 삭제
        os.remove(file_path)
        print(f'Deleted: {file_path}')


# 사용 예시
folder_path = 'C:\\Users\AIA\Desktop\교차로 데이터\(2차_최종) 교차로정보 데이터셋_20210720\횡단보도 이미지'  # 여기 폴더 경로를 입력하세요
delete_files_except_matching_txt(folder_path)
