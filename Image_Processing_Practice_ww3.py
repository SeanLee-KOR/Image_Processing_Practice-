# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 22:17:11 2025

@author: Sean Lee
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

img = cv.imread('scenery.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')


# 부분 선택
# cv.imshow('original_RGB', img)
plt.imshow(img), plt.xticks([]), plt.yticks([]) 
plt.show()
# cv.imshow('Upper left half', img[0:img.shape[0]//2,0:img.shape[1]//2,:])
plt.imshow(img[0:img.shape[0]//2,0:img.shape[1]//2,:]), plt.xticks([]), plt.yticks([]) 
plt.show()
# cv.imshow('Center half',img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[0]//4,:])
plt.imshow(img[img.shape[0]//4:3*img.shape[0]//4, img.shape[1]//4:3*img.shape[0]//4,:]), plt.xticks([]), plt.yticks([]) 
plt.show()

# RGB 채널별로 디스플레이
# cv.imshow('R channel', img[:,:,2])
plt.imshow(img[:,:,2]), plt.xticks([]), plt.yticks([]) 
plt.show()
# cv.imshow('G channel', img[:,:,1])
plt.imshow(img[:,:,1]), plt.xticks([]), plt.yticks([]) 
plt.show()
# cv.imshow('B channel', img[:,:,0])
plt.imshow(img[:,:,0]), plt.xticks([]), plt.yticks([]) 
plt.show()

# cv.waitKey()
# cv.destroyAllWindows()


# 오츄 알고리즘

t, bin_img = cv.threshold(img[:,:,2],0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
print('오추 알고리즘이 찾은 최적의 임계값=', t)

# cv.imshow('R channel', img[:,:,2])

plt.imshow(img[:,:,2], cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

# cv.imshow('R channel binarization',bin_img)

plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

# cv.waitKey()
# cv.destroyAllWindows()

# 모폴로지
# 팽창과 침식 연산

plt.imshow(bin_img, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

b = bin_img[bin_img.shape[0]//3+90:bin_img.shape[0]//3+200,bin_img.shape[1]//3:bin_img.shape[1]//3+150]

plt.imshow(b, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

se = np.uint8([[0,0,1,0,0],                            # 구조 요소
               [0,1,1,1,0],
               [1,1,1,1,1],
               [0,1,1,1,0],
               [0,0,1,0,0]])


# 팽창
b_dilation = cv.dilate(b,se,iterations=1)
plt.imshow(b_dilation, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()


# 침식
b_erosion=cv.erode(b,se,iterations=1)
plt.imshow(b_erosion, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()


# 닫힘
b_closing = cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1)
plt.imshow(b_closing, cmap='gray'), plt.xticks([]), plt.yticks([])
plt.show()

# 감마 보정
resize_img = cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25)

def gamma(f, gamma=1.0):
    f1=f/255.0
    return np.uint8(255*(f1**gamma))

gc=np.hstack((gamma(resize_img,0.5),
              gamma(resize_img,0.75),
              gamma(resize_img,1.0),
              gamma(resize_img,2.0),
              gamma(resize_img,3.0)))

# cv.imshow('gamma',gc)
# cv.waitKey()
# cv.destroyAllWindows()

plt.imshow(gc), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

# 히스토그램 평활화

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

h = cv.calcHist([gray],[0],None,[256],[0,256]) # 히스토그램 계산
plt.plot(h, color='r', linewidth=1)
plt.show()

equal = cv.equalizeHist(gray)
plt.imshow(equal, cmap='gray'), plt.xticks([]), plt.yticks([]) # 영상 평활화 출력
plt.show()

h_e = cv.calcHist([equal],[0],None,[256],[0,256]) # 평활화 히스토그램 계산
plt.plot(h_e, color='r', linewidth=1)
plt.show()

# 컨볼루션

img_resize = cv.resize(img,dsize=(0,0), fx=0.4, fy=0.4)
img_gray = cv.cvtColor(img_resize,cv.COLOR_BGR2GRAY)
cv.putText(gray,'Scenery',(10,20) ,cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
# cv.imshow('Original',img_gray)

plt.imshow(img_gray, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

smooth = np.hstack((cv.GaussianBlur(img_gray,(5,5), 0.0),
                    cv.GaussianBlur(img_gray,(9,9), 0.0),
                    cv.GaussianBlur(img_gray,(15,15), 0.0)))
# cv.imshow('Smooth',smooth)
plt.imshow(smooth, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

femboss = np.array([[-1.0, 0.0, 0.0],
                    [ 0.0, 0.0, 0.0],
                    [ 0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128, 0, 255))
emboss_bad = np.uint8(cv.filter2D(gray16, -1, femboss)+128)
emboss_worse = cv.filter2D(gray, -1, femboss)

# cv.imshow('Emboss',emboss)
# cv.imshow('Emboss_bad',emboss_bad)
# cv.imshow('Emboss_worse',emboss_worse)

plt.imshow(emboss, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()
plt.imshow(emboss_bad, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()
plt.imshow(emboss_worse, cmap='gray'), plt.xticks([]), plt.yticks([]) # 흑백출력
plt.show()

# cv.waitKey()
# cv.destroyAllWindows()

# 영상 보간

patch = img[400:500, 170:270, :]

img = cv.rectangle(img, (170,400), (270,500), (255,0,0), 3)
patch1 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation = cv.INTER_NEAREST)
patch2 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation = cv.INTER_LINEAR)
patch3 = cv.resize(patch, dsize=(0,0), fx=5, fy=5, interpolation = cv.INTER_CUBIC)

#cv.imshow('Original',img)
#cv.imshow('Resize nearest',patch1)
#cv.imshow('Resize bilinear',patch2)
#cv.imshow('Resize bicubic',patch3)

plt.imshow(img, cmap='gray'), plt.xticks([]), plt.yticks([]) 
plt.show()
plt.imshow(patch1, cmap='gray'), plt.xticks([]), plt.yticks([]) 
plt.show()
plt.imshow(patch2, cmap='gray'), plt.xticks([]), plt.yticks([]) 
plt.show()
plt.imshow(patch3, cmap='gray'), plt.xticks([]), plt.yticks([]) 
plt.show()

#cv.watiKey()
#cv.destroyAllWindows()

# OpenCV의 시간 효율
import time

def my_cvtGray1(bgr_img):
  g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
  for r in range(bgr_img.shape[0]):
    for c in range(bgr_img.shape[1]):
      g[r,c]=0.114*bgr_img[r,c,0]+0.587*bgr_img[r,c,1]+0.299*bgr_img[r,c,2]
  return np.uint8(g)

def my_cvtGray2(bgr_img):
  g = np.zeros([bgr_img.shape[0], bgr_img.shape[1]])
  g = 0.114*bgr_img[:,:,0]+0.587*bgr_img[:,:,1]+0.299*bgr_img[:,:,2]
  return np.uint8(g)

start = time.time()
my_cvt1 = my_cvtGray1(img)
plt.imshow(my_cvt1), plt.xticks([]), plt.yticks([]) 
plt.show()
print('My time1:', time.time()-start)

start = time.time()
my_cvt2 = my_cvtGray2(img)
plt.imshow(my_cvt2), plt.xticks([]), plt.yticks([]) 
plt.show()
print('My time2:', time.time()-start)

start = time.time()
cv_cvt2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(cv_cvt2), plt.xticks([]), plt.yticks([]) 
plt.show()
print('OpenCV time:', time.time()-start)
