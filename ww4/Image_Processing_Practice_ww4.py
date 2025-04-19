# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 22:39:45 2025

@author: Sean Lee
"""

# 영상처리실제(opencv) 4주차 Edge & Area

# 실습1 Sobel 엣지 검출

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("soccer.jpg")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

grad_x=cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
grad_y=cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)

sobel_x=cv.convertScaleAbs(grad_x)
sobel_y=cv.convertScaleAbs(grad_y)

edge_strength = cv.addWeighted(sobel_x,0.5,sobel_y,0.5,gamma=0)

# 새창을 여는 imshow보다 matplot을 선호함.
plt.imshow(gray), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(sobel_x), plt.title('sobel_x'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(sobel_y), plt.title('sobel_y'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(edge_strength), plt.title('edge_strength'), plt.xticks([]), plt.yticks([])
plt.show()

# 실습2 Canny 엣지 검출

canny1 =cv.Canny(gray,50,150)  # Trow=50, Thigh=150 설정
canny2 =cv.Canny(gray,100,200) # Trow=100, Thigh=200 설정

plt.imshow(gray), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(canny1), plt.title('Canny1'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(canny2), plt.title('Canny2'), plt.xticks([]), plt.yticks([])
plt.show()

# 실습3 경계선 찾기

canny = cv.Canny(gray,100,200)

contour, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

lcontour=[]
for i in range(len(contour)):
    if contour[i].shape[0]>100:
        lcontour.append(contour[i])

cv.drawContours(img,lcontour,-1,(0,255,0),3)

plt.imshow(img), plt.title('Original with contours'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(canny), plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.show()

# 실습4 허프 변환
img_apple = cv.imread("apples.jpg")
img_apple = cv.cvtColor(img_apple,cv.COLOR_RGB2BGR)
gray_apple = cv.cvtColor(img_apple,cv.COLOR_BGR2GRAY)

apples = cv.HoughCircles(gray_apple, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=20,
                         minRadius=50, maxRadius=120)

for i in apples[0]:
    cv.circle(img_apple,(int(i[0]),int(i[1])),int(i[2]),(255,0,0),2)

plt.imshow(img_apple), plt.title('Apple detection'), plt.xticks([]), plt.yticks([])
plt.show()

# 영역 분할
# 실습5 슈퍼 화소 분할

import skimage

img_coffee = skimage.data.coffee()


#img_coffee = cv.cvtColor(img_coffee,cv.COLOR_RGB2BGR)
plt.imshow(img_coffee), plt.title('Coffee image'), plt.xticks([]), plt.yticks([])
plt.show()

slic1 = skimage.segmentation.slic(img_coffee, compactness=20, n_segments=600)
sp_img1 = skimage.segmentation.mark_boundaries(img_coffee,slic1)
sp_img1 = np.uint8(sp_img1*255.0)

slic2 = skimage.segmentation.slic(img_coffee, compactness=40, n_segments=600)
sp_img2 = skimage.segmentation.mark_boundaries(img_coffee,slic2)
sp_img2 = np.uint8(sp_img2*255.0)

plt.imshow(sp_img1), plt.title('Super pixels (compact 20)'), plt.xticks([]), plt.yticks([])
plt.show()
plt.imshow(sp_img2), plt.title('Super pixels (compact 40)'), plt.xticks([]), plt.yticks([])
plt.show()

# 실습6 최적화 분할
import time

start=time.time()
slic = skimage.segmentation.slic(img_coffee, compactness=20, n_segments=600, start_label=1)
g = skimage.graph.rag_mean_color(img_coffee, slic, mode='similarity')
ncut = skimage.graph.cut_normalized(slic,g) # 정규화 절단
print(img_coffee.shape, 'Coffee 영상을 분할하는 데', time.time()-start, '초 소요')

marking = skimage.segmentation.mark_boundaries(img_coffee, ncut)
ncut_coffee = np.uint8(marking*255.0)

plt.imshow(ncut_coffee ), plt.title('Normalized cut'), plt.xticks([]), plt.yticks([])
plt.show()

# 실습7 GrabCut

img = cv.imread("soccer.jpg")
img_show = np.copy(img)

mask = np.zeros((img.shape[0], img.shape[1]),np.uint8)
mask[:,:] = cv.GC_PR_BGD

BrushSiz = 9
LColor, RColor = (255, 0, 0), (0, 0, 255)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)    
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)    
    
    cv.imshow('Painting', img_show)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1) == ord('q'):
        break
    
# GrabCut 적용코드
background = np.zeros((1,65), np.float64) # 배경 히스토그램 0 초기화
foreground = np.zeros((1,65), np.float64) # 물체 히스토그램 0 최기화

cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
mask2 = np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD),0,1).astype('uint8')
grab = img*mask2[:,:,np.newaxis]
cv.imshow('Grab cut image', grab)

cv.waitKey()
cv.destroyAllWindows()   


# 실습8 영역 특징

orig = skimage.data.horse()
img = 255 - np.uint8(orig) * 255
cv.imshow('Horse',img)

contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.drawContours(img2, contours, -1, (255,0,255),2)
cv.imshow('Horse with contour', img2)

contour = contours[0]

m = cv.moments(contour)
area = cv.contourArea(contour)
cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
perimeter = cv.arcLength(contour,True)
roundness = (4.0*np.pi*area)/(perimeter*perimeter)
print('면적=', area, '\n중점=(',cx,',',cy,')','\n둘레=', perimeter, '\n둥근 정도=', roundness)

img3 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

contour_approx = cv.approxPolyDP(contour, 8, True)
cv.drawContours(img3, [contour_approx], -1, (0,255,0), 2)

hull = cv.convexHull(contour)
hull=hull.reshape(1,hull.shape[0],hull.shape[2])
cv.drawContours(img3, hull, -1, (0,0,255), 2)

cv.imshow('Horse with line segments and convex hull', img3)

cv.waitKey()
cv.destroyAllWindows()