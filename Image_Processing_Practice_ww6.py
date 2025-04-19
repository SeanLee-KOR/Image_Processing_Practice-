# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 00:40:11 2025

@author: Sean Lee
"""

from PyQt5.QtWidgets import *
import sys
import winsound
import cv2 as cv
import numpy as np

# 실습 1(비프음)
class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('삑 소리 내기')
        self.setGeometry(200, 200, 500, 100)

        # 버튼 생성
        shortBeepButton = QPushButton('짧게 삑', self)
        longBeepButton = QPushButton('길게 삑', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        # 버튼 위치 설정
        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        # 버튼 클릭 시 연결될 함수
        shortBeepButton.clicked.connect(self.shortBeepFunction)
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    def shortBeepFunction(self):
        winsound.Beep(1000, 200)  # 1000Hz로 0.2초 삑 소리
        self.label.setText('짧은 삑!')

    def longBeepFunction(self):
        winsound.Beep(1000, 1000)  # 1000Hz로 1초 삑 소리
        self.label.setText('긴 삑!')

    def quitFunction(self):
        self.close()
'''
app = QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('비프음')
win = BeepSound()
win.show()
app.exec_()
'''

# 실습 2(비디오프레임 저장)
class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200, 200, 500, 100)

        # 버튼 생성
        videoButton = QPushButton('비디오 켜기', self)
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        # 버튼 위치 및 크기 설정
        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        # 콜백 함수 연결
        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened(): self.close()

        while True:
            ret, self.frame = self.cap.read()
            if not ret: break
            cv.imshow('Live display', self.frame)
            cv.waitKey(1)

    def captureFunction(self):
        self.captured_frame = self.frame.copy()
        cv.imshow('Captured Frame', self.captured_frame)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self,'파일 저장', './')
        cv.imwrite(fname[0], self.captured_frame)

    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

'''
app = QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('비디오프레임 저장')
win = Video()
win.show()
app.exec_()
'''

# 실습 3(이미지 오림)
class Orim(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200, 200, 700, 200)
        
        fileButton = QPushButton('파일', self)
        paintButton = QPushButton('페인팅', self)
        cutButton = QPushButton('오림', self)
        incButton = QPushButton('+', self)
        decButton = QPushButton('-', self)
        saveButton = QPushButton('저장', self)
        quitButton = QPushButton('나가기', self)

        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(360, 10, 50, 30)
        saveButton.setGeometry(410, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)

        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.decFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.BrushSiz = 5  # 붓 크기
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)  # 파란색 물체, 빨간색 배경

    def fileOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        self.img = cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')

        self.img_show = np.copy(self.img)
        cv.imshow('Painting', self.img_show)
        
        self.mask = np.zeros(self.img.shape[:2], np.uint8)
        self.mask[:, :] = cv.GC_PR_BGD  # 모든 화소를 배경일 것 같음으로 초기화

    def paintFunction(self):
        cv.setMouseCallback('Painting', self.painting)

    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_FGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.BrushSiz, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.BrushSiz, cv.GC_BGD, -1)

        cv.imshow('Painting', self.img_show)

    def cutFunction(self):
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)

        mask2 = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]
        cv.imshow('Scissoring', self.grabImg)

    def incFunction(self):
        self.BrushSiz = min(20, self.BrushSiz+1)

    def decFunction(self):
        self.BrushSiz = max(1, self.BrushSiz-1)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        cv.imwrite(fname[0], self.grabImg)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

'''
app = QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('이미지 오림')
ex = Orim()
ex.show()
app.exec_()
'''

# 실습 4(교통약자 보호구역 알림)
class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('교통약자 보호')
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton('표지판 등록', self)
        roadButton = QPushButton('도로 영상 불러옴', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('나가기', self)
        self.label = QLabel('환영합니다!', self)

        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [['child.png', '어린이'], ['elder.png', '노인'], ['disabled.png', '장애인']]
        self.signImgs = []

    def signFunction(self):
        self.label.clear()
        self.label.setText('교통약자 표지판을 등록합니다.')

        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname, self.signImgs[-1])

    def roadFunction(self):
        if self.signImgs == []:
            self.label.setText('먼저 표지판을 등록하세요.')
        else:
            fname = QFileDialog.getOpenFileName(self, '파일 읽기', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None: sys.exit("파일을 찾을 수 없습니다.")
            cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText('먼저 도로 영상을 입력하세요.')
        else:
            sift = cv.SIFT.create() # cv 버전 주의 cv.SIFT_create()
            KD = []  # 키포인트와 기술자 저장

            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))

            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)

            matcher = cv.DescriptorMatcher.create(cv.DescriptorMatcher_FLANNBASED) # cv 버전 주의 cv.DescriptorMatcher_create()
            GM=[]

            for sign_kp, sign_des in KD:
                knn_match = matcher.knnMatch(sign_des, road_des, k=2)
                T=0.7
                good_match = []
                for nearest1, nearest2 in knn_match:
                    if (nearest1.distance/nearest2.distance) < T:
                        good_match.append(nearest1)
                GM.append(good_match)
                
            best = GM.index(max(GM,key=len)) # 매치쌍이 최대인 표지판 찾기
            
            best_kp, best_des = KD[best]
            best_match = GM[best]
            
            # 시각화를 위한 매칭 이미지 생성
            img_match = cv.drawMatches(self.signImgs[best], best_kp, self.roadImg, road_kp, best_match, None,
                                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imshow('Matches and Homography', img_match)
            
            self.label.setText(self.signFiles[best][1]+ '보호구역입니다. 30km로 서행하세요.')
            winsound.Beep(3000, 500)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()
'''
app = QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('교통약자 보호구역 알림')
win = TrafficWeak()
win.show()
app.exec_()
'''

# 실습 5(파노라마 영상 제작)

class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("파노라마 영상")
        self.setGeometry(200, 200, 700, 200)

        collectButton = QPushButton("영상 수집", self)
        self.showButton = QPushButton("영상 보기", self)
        self.stitchButton = QPushButton("봉합", self)
        self.saveButton = QPushButton("저장", self)
        quitButton = QPushButton("나가기", self)
        self.label = QLabel("환영합니다",self)

        collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        quitButton.setGeometry(450, 25, 100, 30)
        self.label.setGeometry(10, 70, 600, 170)

        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def collectFunction(self):
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText("c를 여러 번 눌러 수집하고 끝나면 q를 눌러 비티오를 끕니다")
        
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened(): sys.exit("카메라 연결 실패")

        self.imgs = []
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            cv.imshow('Video display', frame)
            
            key = cv.waitKey(1)

            if key == ord('c'):
                self.imgs.append(frame) #영상 획득
            elif key == ord('q'):
                self.cap.release()
                cv.destroyWindow('Video display')
                break

        if len(self.imgs)>=2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)

    def showFunction(self):
        self.label.setText('수집된 영상은 '+str(len(self.imgs))+'장입니다.')
        stack=cv.resize(self.imgs[0],dsize=(0,0),fx=0.25,fy=0.25)
        for i in range(1,len(self.imgs)):
            stack=np.hstack((stack,cv.resize(self.imgs[i],dsize=(0,0),fx=0.25,
            fy=0.25)))
        cv.imshow('Image collection',stack)
    
    def stitchFunction(self):
        stitcher=cv.Stitcher_create()
        status,self.img_stitched=stitcher.stitch(self.imgs)
        if status==cv.STITCHER_OK:
            cv.imshow('Image stitched panorama',self.img_stitched)
        else:
            winsound.Beep(3000,500)
            self.label.setText('파노라마 제작에 실패했습니다. 다시 시도하세요.')

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        cv.imwrite(fname[0], self.img_stitched)
    
    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()
'''
app = QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('파노라마 영상 제작')
win = Panorama()
win.show()
app.exec_()
'''

# 실습 6(사진 특수효과 처리)

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('사진 특수 효과')
        self.setGeometry(200,200,800,200)

        pictureButton=QPushButton('사진 읽기',self)
        embossButton=QPushButton('엠보싱',self)
        cartoonButton=QPushButton('카툰',self)
        sketchButton=QPushButton('연필 스케치',self)
        oilButton=QPushButton('유화',self)
        saveButton=QPushButton('저장하기',self)
        self.pickCombo=QComboBox(self)
        self.pickCombo.addItems(['엠보싱','카툰','연필 스케치(명암)','연필 스케치(컬러)','유화'])
        quitButton=QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self)
        
        pictureButton.setGeometry(10,10,100,30)
        embossButton.setGeometry(110,10,100,30)
        cartoonButton.setGeometry(210,10,100,30)
        sketchButton.setGeometry(310,10,100,30)
        oilButton.setGeometry(410,10,100,30)
        saveButton.setGeometry(510,10,100,30)
        self.pickCombo.setGeometry(510,40,110,30)
        quitButton.setGeometry(620,10,100,30)
        self.label.setGeometry(10,40,500,170)
        
        pictureButton.clicked.connect(self.pictureOpenFunction)
        embossButton.clicked.connect(self.embossFunction)
        cartoonButton.clicked.connect(self.cartoonFunction)
        sketchButton.clicked.connect(self.sketchFunction)
        oilButton.clicked.connect(self.oilFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)
        
    def pictureOpenFunction(self):
        fname=QFileDialog.getOpenFileName(self,'사진 읽기','./')
        self.img=cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')
    
        cv.imshow('Painting',self.img)
    
    def embossFunction(self):
        femboss=np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]])
    
        gray=cv.cvtColor(self.img,cv.COLOR_BGR2GRAY)
        gray16=np.int16(gray)
        self.emboss=np.uint8(np.clip(cv.filter2D(gray16,-1,femboss)+128,0,255))
    
        cv.imshow('Emboss',self.emboss)
    
    def cartoonFunction(self):
        self.cartoon=cv.stylization(self.img,sigma_s=60,sigma_r=0.45)
        cv.imshow('Cartoon',self.cartoon)
    
    def sketchFunction(self):
        self.sketch_gray,self.sketch_color=cv.pencilSketch(self.img,sigma_s=60,sigma_r=0.07,shade_factor=0.02)
        cv.imshow('Pencil sketch(gray)',self.sketch_gray)
        cv.imshow('Pencil sketch(color)',self.sketch_color)
        
    def oilFunction(self):
        self.oil=cv.xphoto.oilPainting(self.img,10,1,cv.COLOR_BGR2Lab)
        cv.imshow('Oil painting',self.oil)
    
    def saveFunction(self):
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')
    
        i=self.pickCombo.currentIndex()
        if i==0: cv.imwrite(fname[0],self.emboss)
        elif i==1: cv.imwrite(fname[0],self.cartoon)
        elif i==2: cv.imwrite(fname[0],self.sketch_gray)
        elif i==3: cv.imwrite(fname[0],self.sketch_color)
        elif i==4: cv.imwrite(fname[0],self.oil)
    
    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()

app=QApplication(sys.argv)
window = QWidget()            
window.setWindowTitle('사진 특수효과 처리')
win=SpecialEffect()
win.show()
app.exec_()