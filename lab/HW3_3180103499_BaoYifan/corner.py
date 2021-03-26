# -*- coding: utf-8 -*-
# autho: Bao Yifan
# Harris Corner Detection Implementation
# Date: 2020.12.18
import cv2
import numpy as np

# 正则化到0到255浮点数
def normalization(data):
    theRange = np.max(data) - np.min(data)
    return ((data - np.min(data)) / theRange) * 255

# harris corner代码实现
def HarrisCorner(img, ksize=3):
    k = 0.04  # 经验值k-响应函数计算公式中
    threshold = 0.01 # threshold 阈值 用于选择corner
    # 采用NMS非极大值抑制 会导致最终选择到的点过少 -- 这里没有采用
    height, width = img.shape[:2]  # 获取图像高和宽
    # 使用sobel算子计算两个方向的梯度
    grad = np.zeros((height,width,2),dtype=np.float32)
    # 注意这里采用cv2.CV_16S主要是因为梯度计算的值有负数
    grad[:, :, 0] = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3) # x方向梯度
    grad[:, :, 1] = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3) # y方向梯度
    # m矩阵相关参数计算//四个角的值
    m = np.zeros((height,width,3),dtype=np.float32)
    m[:, :, 0] = grad[:, :, 0]**2 # Ix^2,
    m[:, :, 1] = grad[:, :, 1]**2 # Iy^2
    m[:, :, 2] = grad[:, :, 0]*grad[:, :, 1] # Ix*Iy 
    # m矩阵计算 利用高斯卷积
    m[:,:,0] = cv2.GaussianBlur(m[:,:,0],ksize=(ksize,ksize),sigmaX=2)    
    m[:,:,1] = cv2.GaussianBlur(m[:,:,1],ksize=(ksize,ksize),sigmaX=2)
    m[:,:,2] = cv2.GaussianBlur(m[:,:,2],ksize=(ksize,ksize),sigmaX=2)  
    #计算得m矩阵  
    m = [np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(height) for j in range(width)]
    # 根据M矩阵 计算特征值---用于完成中间步骤图的绘制
    value,vector = np.linalg.eig(m)
    value.sort()
    # 计算M矩阵的行列式 与 迹
    D,T = list(map(np.linalg.det,m)),list(map(np.trace,m))
    # 根据结果计算响应函数R---R(i,j)=det(M)-k(trace(M))^2
    R = np.array([d-k*t**2 for d,t in zip(D,T)])

    # 阈值获取角点
    R_max = np.max(R)   
    R = R.reshape(height,width)

    R_norm = normalization(R)
    # R_norm[R_norm<150] = 0
    # R_norm[R_norm>=150]=255
    cv2.imshow('Rvalue',R_norm.astype(np.uint8))
    cv2.imwrite('Rvalue.jpg',R_norm.astype(np.uint8))

    corner = np.zeros_like(R,dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if R[i,j] > R_max*threshold :
                corner[i,j]=1 # flag 表明是大于阈值的corner
    # 寻找局部极大值 设置角点值为255
    for i in range(1, height-1):
        for j in range(1, width-1):
            if corner[i,j]==1 and R[i][j] > R[i-1][j] and R[i][j] > R[i+1][j] and R[i][j] > R[i][j+1] and R[i][j] > R[i][j-1]:
                corner[i,j]=255 # 满足局部极大值 设置为255
    return value, corner
    
if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    while(1):
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.waitKey(0)
            # #转换为灰度图像
            img = frame
            imgShape = img.shape
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            value, dst = HarrisCorner(gray)
            minValue = value[:,0]  #最小特征值
            maxValue = value[:,1]  #最大特征值
            minValue = np.reshape(minValue,(imgShape[0],imgShape[1]))
            maxValue = np.reshape(maxValue,(imgShape[0],imgShape[1]))

            minValue=minValue.astype(np.uint8)
            maxValue=maxValue.astype(np.uint8)
            cv2.imshow('minEigenvalues',minValue)
            cv2.imwrite('minEigenvalues.jpg',minValue)
            cv2.imshow('maxEigenvalues',maxValue)
            cv2.imwrite('maxEigenvalues.jpg',maxValue)
            frame[dst>0.01*dst.max()] = [0,0,255]
            cv2.imshow('result',frame)
            cv2.imwrite('result.jpg',frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        #cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows() 
  
    
