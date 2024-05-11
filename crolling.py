import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import os

# 이미지를 저장할 폴더 생성
if not os.path.exists('images'):
    os.makedirs('images')

# 검색 URL
url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=image&ssc=tab.image.all&query=%EC%95%88%EC%A0%84%EC%A0%9C%EC%9D%BC+%ED%9C%80%EC%8A%A4&oquery=%EC%95%88%EC%A0%84%EC%A0%9C%EC%9D%BC+%ED%91%9C%EC%A7%80%ED%8C%90&tqi=iBvnrsqVN8wssTgy9KGssssstzR-200762'

# 웹 드라이버 시작
driver = webdriver.Chrome()

# 페이지 열기
driver.get(url)

# 스크롤을 내려 더 많은 이미지 로드
for i in range(10):
    # 스크롤 내리는 시간
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# 페이지 소스 가져오기
page_source = driver.page_source

# 웹 드라이버 종료
driver.quit()

# BeautifulSoup 객체 생성
soup = BeautifulSoup(page_source, 'html.parser')

# 이미지 태그 추출
image_tags = soup.find_all('img')

# 이미지 다운로드 및 저장
count = 0
for i, image_tag in enumerate(image_tags):
    try:
        # 이미지 URL 추출
        image_url = image_tag['src']

        # 이미지 다운로드
        image_data = requests.get(image_url).content

        # 이미지 저장
        with open(f'images/안전제일_{i}.jpg', 'wb') as f:
            f.write(image_data)

        print(f'이미지 {i} 다운로드 및 저장 완료')
        count += 1
    except Exception as e:
        print(f'이미지 {i} 다운로드 실패:', e)

    # 이미지가 200개를 넘으면 종료
    if count >= 200:
        break

    # 서버 부하 방지를 위해 1초 대기
    time.sleep(1)
