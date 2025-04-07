# 영상처리 실제 - 지역 특징
# 대응점 문제

import numpy as np
import cv2 as cv
import time

'''
'''
# 구분      | 지역 특징           | 영역 특징
# 범위      | 작은 패치, 점 단위   | 넓은 영역, 객체 단위
# 추출 방식 | 특징점 기반          | 영역 제안 또는 Rol 기반 
# 표현 방식 | 디스크립터           | CNN 피처맵에서 추출된 텐서
# 용도      | 정합, 추적, 구조 추정| 객체 인식, 분할, 탐지
# 분포      | 희소(sparse)        | 조밀하거나 영역 중심

# 반복성
# 불변성
# 분별력
# 지역성
# 적당한 양
# 계산 효율

# 모라벡의 설명
# 해리스 특징점 - Gausian Kernel 활용
k = cv.getGaussianKernel(5,1)
print(k)

# 스케일 불변한 지역특징

# 가우시안 스무딩 - 스무딩
# 피라미드 방법 - 스케일을 줄임
# 정규라플라시안
# Defferance of Gausian

# SIFT
# 특징점 검출: SURF, FAST ,AGAST 등
# 기술자 추출: PCA-SIFT, GLOH, 모양 컨텍스트, 이진 기술자 등

# 매칭 전략
# 두 기술자 간 거리의 계산
# 고정 임계값 d(A,B) = ||a-b||
# 최근접 이웃 d(Ai, Bj) < T
# 최근접 이웃 거리 비율 d(Ai, Bj)/d(Ai, Bk) < T
# 매칭 성능 측정 - confusion matrix, ROC, AUC

# 호모그래피 추정

