#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:14:00 2018

@author: anildogan
"""
import sys
from PyQt5.QtWidgets import QMainWindow,QMessageBox, QApplication,QScrollArea, QWidget, QPushButton, QAction, QGroupBox, QFileDialog, QLabel, QVBoxLayout, QGridLayout, QHBoxLayout,QFrame, QSplitter,QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap, QPalette,QImage
from PyQt5.QtCore import pyqtSlot, Qt
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import math
from numpy.linalg import inv
from scipy import signal
from PIL import Image
import random
class App(QMainWindow):
    
    def __init__(self):
        super(App,self).__init__()
        
        self.window = QWidget(self)
        self.setCentralWidget(self.window)
    
        self.inputBox = QGroupBox('Input')
        inputLayout = QVBoxLayout()
        self.inputBox.setLayout(inputLayout)
        
        
        self.resultBox = QGroupBox('Result')
        resultLayout = QVBoxLayout()
        self.resultBox.setLayout(resultLayout)
        
        self.layout = QGridLayout()
        self.layout.addWidget(self.inputBox, 0, 0)
        self.layout.addWidget(self.resultBox, 0, 1)
        
        self.window.setLayout(self.layout)
        
        self.image = None
        self.segmentImage = None
        self.gray = None
        self.result = None
        self.showimage = None
        self.filtered = None
        self.filtered1 = None
        self.filtered2 = None
        self.tmp_im = None
        self.imageLabel = None
        self.imageLabel2 = None
        self.figure = Figure()
        self.figure2 = Figure()
        self.figure3 = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas2 = FigureCanvas(self.figure2)
        self.canvas3 = FigureCanvas(self.figure3)
        self.lookupRed = np.zeros((256,1))
        self.lookupGreen = np.zeros((256,1))
        self.lookupBlue = np.zeros((256,1))
        self.arnoldpoints = None
        self.bushpoints = None
        self.eq = None
        self.qImg = None
        self.qImg2 = None
        self.qImgResult = None
        self.pixmap01 = None
        self.pixmap_image = None
        self.delaunay_color = (255,0,0)
        
        self.createActions()
        self.createMenu()
        self.createToolBar()
        
        self.setWindowTitle("Histogram")
        self.showMaximized()
        self.show()
        
        
    def createActions(self):
        self.open_inputAct = QAction(' &Open Input',self)
        self.open_inputAct.triggered.connect(self.open_Input)
        self.exitAct = QAction(' &Exit', self)
        self.exitAct.triggered.connect(self.exit)
        self.harris = QAction(' &Harris Corner Detection',self)
        self.harris.triggered.connect(self.harrisFunc)
        self.segment = QAction(' &Segmentation',self)
        self.segment.triggered.connect(self.segmentFunc)
    
    def createMenu(self):
        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('File')
        self.fileMenu.addAction(self.open_inputAct)
        self.fileMenu.addAction(self.exitAct)
    def createToolBar(self):
        self.har = self.addToolBar("Harris Corner")
        self.har.addAction(self.harris)
        self.seg = self.addToolBar("Segmentation")
        self.seg.addAction(self.segment)
    def sobelFilterX(self):
        h,w=self.gray.shape
 
        filt=3
        summary = 0
        self.filtered1 =self.filtered.copy()
        x= int(filt/2)
        for i in range(0,h):
            for j in range(0,w):
                if(i>=x and j>=x and i<=h-1-x and j<=w-1-x):
                    for a in range(0,filt):
                        for b in range(0,filt):
                            #summary= summary - (a%2+1)*math.sin(90*(b+1))*self.filtered[i-x+b][j-x+a]
                            summary= summary - math.sin(90*(b+1))*self.filtered[i-x+b][j-x+a]
                    self.filtered1[i][j]= summary
                    summary = 0
                else:
                    for a in range(0,filt):
                        for b in range(0,filt):
                            if(i-x+b<0 or i-x+b>h-1 or j-x+a<0 or j-x+a>w-1):
                                x = x
                            else:
                                #summary= summary -(a%2+1)*math.sin(90*(b+1))*self.filtered[i-x+b][j-x+a]
                                summary= summary -math.sin(90*(b+1))*self.filtered[i-x+b][j-x+a]
                                
                    self.filtered1[i][j] = summary
                    summary = 0  
                    
        
    def sobelFilterY(self):
        h,w=self.gray.shape
 
        filt=3
        summary = 0
        self.filtered2 =self.filtered.copy()
        x= int(filt/2)
        for i in range(0,h):
            for j in range(0,w):
                if(i>=x and j>=x and i<=h-1-x and j<=w-1-x):
                    for a in range(0,filt):
                        for b in range(0,filt):
                            #summary= summary - (b%2+1)*math.sin(90*(a+1))*self.filtered[i-x+b][j-x+a]
                            summary= summary - math.sin(90*(a+1))*self.filtered[i-x+b][j-x+a]
                    self.filtered2[i][j]= summary
                    summary = 0
                else:
                    for a in range(0,filt):
                        for b in range(0,filt):
                            if(i-x+b<0 or i-x+b>h-1 or j-x+a<0 or j-x+a>w-1):
                                x = x
                            else:
                               # summary= summary -(b%2+1)*math.sin(90*(a+1))*self.filtered[i-x+b][j-x+a]
                               summary= summary -math.sin(90*(a+1))*self.filtered[i-x+b][j-x+a]
                                
                    self.filtered2[i][j] = summary
                    summary = 0
    def gaussianFilters(self):
        h,w=self.gray.shape
        s = 1
        firstx = 1/(2*math.pi*s)
        sums = 0
        filt=3
        summary = 0
        self.filtered =self.image.copy()
        x= int(filt/2)
        for i in range(0,h):
            for j in range(0,w):
                if(i>=x and j>=x and i<=h-1-x and j<=w-1-x):
                    for a in range(0,filt):
                        for b in range(0,filt):
                            sums = sums + firstx*math.exp(-1*((-x+b)*(-x+b)+(-x+a)*(-x+a))/(2*s))
                            summary= summary + self.image[i-x+b][j-x+a]*(firstx*math.exp(-1*((-x+b)*(-x+b)+(-x+a)*(-x+a))/(2*s)))
                    self.filtered[i][j]= summary/sums
                    summary = 0
                    sums = 0
                else:
                    for a in range(0,filt):
                        for b in range(0,filt):
                            if(i-x+b<0 or i-x+b>h-1 or j-x+a<0 or j-x+a>w-1):
                                x = x
                            else:
                                sums = sums + firstx*math.exp(-1*(((-x+b)*(-x+b)+(-x+a)*(-x+a))/(2*s)))
                                summary= summary + self.image[i-x+b][j-x+a]*(firstx*math.exp((-1)*((-x+b)*(-x+b)+(-x+a)*(-x+a))/(2*s)))
                                
                    self.filtered[i][j] = summary/sums
                    summary = 0
                    sums = 0
    def gradientx(self):
        h,w=self.gray.shape
 
        summary = 0
        self.filtered1 =self.filtered.copy()
        for i in range(0,h):
            for j in range(0,w):
                if(j>=1 and j<=w-2):
                    summary = - self.filtered[i][j-1] + self.filtered[i][j+1]
                    self.filtered1[i][j]= summary
                    summary = 0
                else:     
                    if j==0:
                        summary= self.filtered[i][j+1]
                    elif j==w-1:
                        summary= self.filtered[i][j-1]
                                
                    self.filtered1[i][j] = summary
                    summary = 0
    def gradienty(self):
        h,w=self.gray.shape
 
        summary = 0
        self.filtered2 =self.filtered.copy()
        for i in range(0,h):
            for j in range(0,w):
                if(i>=1 and i<=h-2):
                    summary = - self.filtered[i-1][j] + self.filtered[i+1][j]
                    self.filtered2[i][j]= summary
                    summary = 0
                else:     
                    if i==0:
                        summary= self.filtered[i+1][j]
                    elif i==h-1:
                        summary= self.filtered[i-1][j]
                                
                    self.filtered2[i][j] = summary
                    summary = 0   
    def segmentFunc(self):
        print("segment")
        self.segmentImage=self.image.copy()
        originalImage =self.image.copy()
        self.segmentImage = cv2.cvtColor(self.segmentImage, cv2.COLOR_BGR2GRAY)
        
        height,width = self.segmentImage.shape
        
        brain = np.array(self.segmentImage)
        kmeans = self.segmentImage.copy()
        hist = np.zeros(256)
        for i in range(height):
            for j in range(width):
                hist[brain[i][j]] += 1

        thresh = 120
        
        for i in range(height):
            for j in range(width):
                if(brain[i][j] < thresh):
                    brain[i][j] = 0
                else:
                    brain[i][j] = 255
        
        array = np.array(brain, dtype=np.uint8)
        t = Image.fromarray(array)
        t.save('mrthresh.jpg')
        
        kernel = np.ones((12,12),np.uint8)
        
        erosion = cv2.erode(array,kernel,iterations = 1)
        erosion = np.array(erosion, dtype=np.uint8)
        t = Image.fromarray(erosion)
        t.save('skull.jpg')
        
        mean1 = random.randint(0,256)
        mean2 = random.randint(0,256)
        lst1 = []
        lst2 = []
        lst1index = []
        lst2index = []
        prevmean1 = 0
        prevmean2 = 0
        while(prevmean1!=mean1 and prevmean2!=mean2):
            lst1index = []
            lst2index = []
            for i in range(height):
                for j in range(width):
                    if(erosion[i][j]==255):
                        dif1 = self.segmentImage[i][j]-mean1
                        dif2 = self.segmentImage[i][j]-mean2
                        if(abs(dif1)<abs(dif2)):
                            lst1.append(originalImage[i][j])
                            lst1index.append((i,j))
                        else:
                            lst2.append(originalImage[i][j]) 
                            lst2index.append((i,j))

            try:
                newmean1 = int(np.mean(lst1))
                newmean2 = int(np.mean(lst2))
                string1 = "prev1 = " + str(prevmean1) + "   mean1 = " + str(mean1)
                string2 = "prev2 = " + str(prevmean2) + "   mean2 = " + str(mean2)
                print(string1)
                print(string2)
                prevmean1 = mean1
                prevmean2 = mean2
                mean1 = newmean1
                mean2 = newmean2
                
            except:
                print("except")
                mean1 = random.randint(0,256)
                mean2 = random.randint(0,256)
    
        for i in range(height):
            for j in range(width):
                if((i,j) in lst1index):
                    if(len(lst1index)>=len(lst2index)):
                        kmeans[i][j] = 127
                    else:
                        kmeans[i][j] = 255
                    
                elif((i,j) in lst2index):
                    if(len(lst1index)>=len(lst2index)):
                        kmeans[i][j] = 255
                    else:
                        kmeans[i][j] = 127
                else:
                    kmeans[i][j]= 0
        tumor = np.array(kmeans, dtype=np.uint8)
        s = Image.fromarray(tumor)
        s.save('tumor.jpg')
        
        
        onlytumor = np.zeros(tumor.shape)
        
        for i in range(height):
            for j in range(width):
                if(tumor[i][j]==255):
                    onlytumor[i][j]=255
        
        onlytumor = cv2.erode(onlytumor,kernel,iterations = 1)   
        onlytumor = cv2.dilate(onlytumor,kernel,iterations = 1) 
        boundaryTumor = cv2.morphologyEx(onlytumor, cv2.MORPH_GRADIENT, kernel)
        for i in range(height):
            for j in range(width):
                if(boundaryTumor[i][j] == 255):
                    originalImage[i][j][0] = 255
                    originalImage[i][j][1] = 0
                    originalImage[i][j][2] = 0
        
        
        
        self.qImg2 = QImage(originalImage.data,width,height,3*width,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel2 = QLabel('result')
        self.imageLabel2.setPixmap(QPixmap.fromImage(self.qImg2))
        self.imageLabel2.setAlignment(Qt.AlignCenter)
        self.resultBox.layout().addWidget(self.imageLabel2)
    def harrisFunc(self,thresh):
        self.gaussianFilters()
        height,width =self.gray.shape
        self.gradientx()
        self.gradienty()
        Ix=self.filtered1
        Iy=self.filtered2
        
        Iyy=np.zeros((self.filtered2.shape),dtype=int)
        Ixx=np.zeros((self.filtered1.shape),dtype=int)
        Ixy=np.zeros((self.filtered1.shape),dtype=int)

        Iyy= Iy*Iy
        Ixx =Ix*Ix
        Ixy= Iy*Ix

        windowsize=3         
        offset = int(windowsize/2)
        for a in range(offset,height-offset):
            for b in range(offset, width-offset):
                
                windowIxx = Ixx[a-offset:a+offset+1,b-offset:b+offset+1]
                windowIyy = Iyy[a-offset:a+offset+1,b-offset:b+offset+1]
                windowIxy = Ixy[a-offset:a+offset+1,b-offset:b+offset+1]
                
                sumxx = windowIxx.sum()
                sumyy = windowIyy.sum()
                sumxy = windowIxy.sum()

                det = (sumxx*sumyy)-(sumxy**2)

                trace = sumxx +sumyy

                k=0.04
                #thresh=0.5
                R = det-k*(trace*trace)
                if R>2500000:
                    self.image[a,b,0]= 0
                    self.image[a,b,1]= 255
                    self.image[a,b,2]= 0
                    print("R buyuktur threshten",R)
        self.qImg2 = QImage(self.image.data,width,height,QImage.Format_RGB888).rgbSwapped()
        
        self.imageLabel2 = QLabel('result')
        self.imageLabel2.setPixmap(QPixmap.fromImage(self.qImg2))
        self.imageLabel2.setAlignment(Qt.AlignCenter)
        self.resultBox.layout().addWidget(self.imageLabel2)
    def open_Input(self):
        #fileName, _ = QFileDialog.getOpenFileName(self, "Open File",QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Input', '.')
        if fileName:
            self.image = cv2.imread(fileName)
            heightI,widthI,dim = self.image.shape
            bytesPerLine = dim*widthI
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.result = np.zeros((heightI,widthI),dtype=int)
            
            if not self.image.data:
                QMessageBox.information(self, "Image Viewer",
                        "Cannot load %s." % fileName)
                return
        self.showimage=self.image.copy()
      #  self.image = self.image[:,:,1]
        self.qImg = QImage(self.showimage.data,widthI,heightI,bytesPerLine,QImage.Format_RGB888).rgbSwapped()
       # self.qImg = QImage(self.showimage.data,widthI,heightI,QImage.Format_Grayscale8)
        self.imageLabel = QLabel('image')
        self.imageLabel.setPixmap(QPixmap.fromImage(self.qImg))
        self.imageLabel.setAlignment(Qt.AlignCenter)
        
        self.inputBox.layout().addWidget(self.imageLabel)
    
    def exit(self):
        sys.exit()
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
