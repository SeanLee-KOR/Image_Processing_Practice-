# 영상처리 실제 - 지역 특징
# 대응점 문제

import numpy as np
import cv2 as cv
import time

'''
지역 특징 차이점
구분      | 지역 특징            | 영역 특징
범위      | 작은 패치, 점 단위    | 넓은 영역, 객체 단위
추출 방식  | 특징점 기반          | 영역 제안 또는 Rol 기반 
표현 방식  | 디스크립터           | CNN 피처맵에서 추출된 텐서
용도      | 정합, 추적, 구조 추정 | 객체 인식, 분할, 탐지
분포      | 희소(sparse)        | 조밀하거나 영역 중심

지역 특징의 조건
 - 반복성
 - 불변성
 - 분별력
 - 지역성
 - 적당한 양
 - 계산 효율

모라벡 알고리즘 - 제곱차의 합 계산
S(0,1) = sigma(2<y<6)sigma(1<x<5)[f(y,x+1)-f(y,x)]^2 = 4
'''

# 해리스 특징점 실습 - Gausian Kernel 활용
k = cv.getGaussianKernel(5,1)
print(k)

img = np.zeros((10,10), dtype=np.float32)
for i in range(5):
    img[2+i][3:4+i] = 1
print(img)

ux = np.array([[-1,0,1]])
uy = np.array([-1,0,1]).transpose()
k = cv.getGaussianKernel(3,1)
g = np.outer(k,k.transpose())

dy = cv.filter2D(img, cv.CV_32F, uy)
print(dy)

dx = cv.filter2D(img, cv.CV_32F, ux)
print(dx)

dyy = dy*dy
print(dyy)

dxx = dx*dx
print(dxx)

dyx = dy*dx
print(dyx)

gdyy = cv.filter2D(dyy,cv.CV_32F,g)
print(gdyy)

gdxx = cv.filter2D(dxx,cv.CV_32F,g)
print(gdxx)

gdyx = cv.filter2D(dyx, cv.CV_32F,g)
print(gdyx)

C = (gdyy*gdxx-gdyx**2) - 0.04*(gdyy+gdxx)**2
print(C)

for i in range(1, C.shape[0]-1):
    for j in range(1, C.shape[1]-1):
        print(C[i,j]>C[i-1:i+2,j-1:j+2]) # Bool 3x3
        print(sum(C[i,j]>C[i-1:i+2,j-1:j+2])) # Sum 3x1
        if C[i,j]>0.1 and sum(sum(C[i,j]>C[i-1:i+2,j-1:j+2])) == 8: # sum sum 1x1 max 8
            img[j,i] = 9 # 중심을 기준으로 주변 8위치가 모두 초과 일때 특징점으로 9로 표시
np.set_printoptions(precision=2)
print(img)

popping = np.zeros([160,160], np.uint8)
for i in range(0,160):
    for j in range(0,160):
        popping[j,i] = np.uint8((C[j//16, i//16]+0.06)*700) 

cv.imshow('Image Display2', popping)
cv.waitKey()
cv.destroyAllWindows()

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

