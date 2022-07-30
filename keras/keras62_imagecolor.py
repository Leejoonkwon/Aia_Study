import cv2
from PIL import Image
path = "/home/test.jpg"
# 이미지 읽기
img_gray = cv2.imread("/home/save_directory/opencv_gary.jpg", cv2.IMREAD_COLOR)

# 컬러 이미지를 그레이스케일로 변환
img_cv_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 이미지 저장
cv2.imwrite(path, img_cv_gray)

# 이미지 사이즈 변경
# img_gray_resize = cv2.resize(img_cv_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 가로, 세로 모두 반으로 줄이기
# img_gray_reduced = cv2.resize(img_cv_gray, None, interpolation=cv2.INTER_AREA)

# 이미지 화면으로 보기
# cv2.imshow('color', img_gray)  # color라는 이름의 윈도우 안에 img_gray 이미지 보여주기
# cv2.imshow('gray-scale', img_gray_resize)
# cv2.imshow('gray-scale reduced', img_gray_reduced)