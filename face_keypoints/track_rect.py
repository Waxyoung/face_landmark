import cv2
import sys
import os
import torch.optim as optim
import torch
import numpy as np
import math

from my_face_dataset import FaceDataSet
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from BigResNet import BigResNet
import sympy
import time
import cv2.cv as cv

left_nose_idx=12
right_nose_idx=20

lk_params = dict( winSize  = (10, 10),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

global img
global point1, point2
global d

file_p0=open('/Users/momo/Downloads/point/point.txt','w+')
file_kalman=open('/Users/momo/Downloads/point/kalman_point.txt','w+')
file_smooth=open('/Users/momo/Downloads/point/smooth_point.txt','w+')



class Kalman2D(object): 
    def __init__(self, processNoiseCovariance=1e-1, measurementNoiseCovariance=1e-1, errorCovariancePost=20):
        self.stateNum=36
        self.measureNum=18
        self.kalman = cv.CreateKalman(self.stateNum, self.measureNum, 0)
        self.kalman_state = cv.CreateMat(self.stateNum, 1, cv.CV_32FC1)
        self.kalman_process_noise = cv.CreateMat(self.stateNum, 1, cv.CV_32FC1)
        self.kalman_measurement = cv.CreateMat(self.measureNum, 1, cv.CV_32FC1)
 
        for i in range(self.stateNum):
            for j in range(self.stateNum):
                if i==j or (j-self.measureNum)==i :
                    self.kalman.transition_matrix[i,j] = 1.0
                else:
                    self.kalman.transition_matrix[i,j] = 0.0 

        cv.SetIdentity(self.kalman.measurement_matrix) 
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(processNoiseCovariance))
        
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(measurementNoiseCovariance))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(errorCovariancePost))
 
        self.predicted = None
        self.esitmated = None

    def update(self,pre):
        pre=pre.reshape(1,-1)
        
        for k in range(self.measureNum):
            self.kalman_measurement[k, 0] = pre[0,k]   

        self.predicted = cv.KalmanPredict(self.kalman)
        self.corrected = cv.KalmanCorrect(self.kalman, self.kalman_measurement)

    def getEstimate(self):
        est=[]
        for m in range(self.measureNum):
            est.append(self.corrected[m,0])
        est=np.array(est)
        est=est.reshape(-1,2)
        return est 
    
    def getPrediction(self): 
        return self.predicted[0,0], self.predicted[1,0]

def checkedTrace(img0, img1, p, back_threshold = 1.0):
    p1r, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1r, None, **lk_params)
    d = abs(p-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return status

def fdis(x):
    if abs(x)>4:
        return x
    return abs(x)*((1-np.exp((-1)*x))/(1+np.exp((-1)*x)))

def fdis1(x):
    if abs(x)>5:
        return x
    return ((1-np.exp((-1)*x))/(1+np.exp((-1)*x)))


def updatePoint(pre,next):
    pre=np.array(pre)
    next=np.array(next)
    for i in range(next.shape[0]):
        # if i>45:
        #     next[i,0]=pre[i,0]+fdis1(next[i,0]-pre[i,0])
        #     next[i,1]=pre[i,1]+fdis1(next[i,1]-pre[i,1])
        # else:
        next[i,0]=pre[i,0]+fdis(next[i,0]-pre[i,0])
        next[i,1]=pre[i,1]+fdis(next[i,1]-pre[i,1])
    return next

def sigmoid(dis):
    return 1/(1+np.exp(-(dis-[4,4])))

def saveLog(p,file):
    p=np.array(p)
    p=p.reshape(1,-1)
    file.write(str(p))
    file.write('\n')

def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         
        point1 = [x,y]
        cv2.circle(img2, (point1[0],point1[1]), 10, (0,255,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):             
        cv2.rectangle(img2, (point1[0],point1[1]), (x,y), (255,0,0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:       
        point2 = [x,y]
        cv2.rectangle(img2, (point1[0],point1[1]), (point2[0],point2[1]), (0,0,255), 5) 
        cv2.imshow('image', img2)
        min_x = min(point1[0],point2[0])     
        min_y = min(point1[1],point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        

lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

def get_small_img_1(img):
    global point1, point2,d
    print point1
    print point2
    d=point2[0]-point1[0]
    ROI=img[point1[1]:point1[1]+d,point1[0]:point1[0]+d,:]
    # ROI=img[3:303,468:768,:]
    dst=float(max(ROI.shape[0],ROI.shape[1]))
    #scale=0.5
    scale=128/dst
    #scale=0.5
    src_middle=[ROI.shape[0]/2,ROI.shape[1]/2]

    rotate_degree=0
    M = cv2.getRotationMatrix2D((src_middle[0], src_middle[1]), rotate_degree, scale)
    #M[:, 2] += offset
    dst_size=128
    dst_middle = np.array([0.5, 0.5]) * dst_size
    offset = dst_middle - src_middle
    M[:, 2] += offset
    img1 = cv2.warpAffine(ROI, M, (dst_size, dst_size))
    # for i in range(label.shape[0]):
    #     cv2.circle(img, (label[i, 0], label[i, 1]), 2, (255, 0, 0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey() 
    return img1,M


def get_small_img_2(img,label):
    global point1, point2

    middle=np.array([point1[0]+d*0.5,point1[1]+d*0.85])
    diff=middle-label[2]
    diff_plus=abs(diff)
    diff_sigmoid=sigmoid(diff_plus)
    diff=np.multiply(diff_sigmoid,diff)
  
  
    point1=np.array(point1-diff).astype(np.int16)
    print point1       
    if point1[1]<0:
        dev=float(abs(point1[0]))
        H=np.array([[1.,0.,0.],[0.,1.,dev]])     
        img= cv2.warpAffine(img, H, (img.shape[1],img.shape[0]))
        label=np.array(label)+np.array([0,dev])
        point1[1]=0
    if point1[0]<0:
        dev=float(abs(point1[0]))    
        H=np.array([[1.,0.,dev],[0.,1.,0]])
        img= cv2.warpAffine(img, H, (img.shape[1],img.shape[0]))
        label=np.array(label)+np.array([dev,0])
             
        point1[0]=0
    
    ROI=img[point1[1]:point1[1]+d,point1[0]:point1[0]+d,:]
    # cv2.rectangle(img, (point1[0],point1[1]), (point1[0]+d,point1[1]+d), (0,0,255), 5)
    
    # ROI=img[3:303,468:768,:]
    
    dst=float(max(ROI.shape[0],ROI.shape[1])) 
    #scale=0.5
    scale=128/dst 
    #scale=0.5
    src_middle_1=[ROI.shape[0]/2,ROI.shape[1]/2] 
    s=np.array(label)-point1

    # src_middle=[(s[0,0]+s[0,1])/2,(s[1,0]+s[1,1])/2]
    src_middle=s[2]
    
    rotate_degree=0
    # rotate_radian = math.atan2( (s[right_nose_idx][1] - s[left_nose_idx][1]) , (s[right_nose_idx][0] - s[left_nose_idx][0]) )
    # rotate_degree = rotate_radian / 3.1415926 * 180
    M = cv2.getRotationMatrix2D((src_middle_1[0], src_middle_1[1]), rotate_degree, scale)
    #M[:, 2] += offset
    dst_size=128
    dst_middle = np.array([0.5, 0.5]) * dst_size
    offset = dst_middle - src_middle_1
    M[:, 2] += offset
    img1 = cv2.warpAffine(ROI, M, (dst_size, dst_size))
    # for i in range(label.shape[0]):
    #     cv2.circle(img, (label[i, 0], label[i, 1]), 2, (255, 0, 0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey() 
    return img1,M


    
    
    # for i in range(label.shape[0]):
    #     cv2.circle(img, (label[i, 0], label[i, 1]), 2, (255, 0, 0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey() 
    

def get_small_img_3(img, label, dst_size = 128,face_scale=0.95):       
    
    src_left_nose = label[left_nose_idx, :]
    src_right_nose = label[right_nose_idx, :]
    src_middle = (src_left_nose + src_right_nose) / 2
    # src_middle = (src_middle + label[2,:]) / 2

    # top_mid = (label[4,:]+label[7,:]) /2

    # disV = cv2.norm(top_mid  - label[2])
    # disH = cv2.norm(label[4]  - label[7])
    # face_size = max(disV*0.95, disH*0.95)

    
    # src_length = min(max(cv2.norm(src_left_nose - src_right_nose), face_size * 0.15), face_size * 0.3)
    src_length = cv2.norm(src_left_nose,src_right_nose)
    dst_middle = np.array([0.5, 0.58]) * dst_size #300:150x174
    dst_length = dst_size * 0.16  #300:54   

    scale = dst_length / src_length * face_scale
    # print 'scale:',scale
    offset = dst_middle - src_middle

    rotate_degree = 0
    rotate_radian = math.atan2( (label[right_nose_idx][1] - label[left_nose_idx][1]) , (label[right_nose_idx][0] - label[left_nose_idx][0]) )
    rotate_degree = rotate_radian / 3.1415926 * 180
  

    M = cv2.getRotationMatrix2D((src_middle[0], src_middle[1]), rotate_degree, scale)
    M[:, 2] += offset
    #img = cv2.warpAffine(img, M, (int(np.math.ceil(128.0)), int(np.math.ceil(128.0))),cv2.INTER_LINEAR,cv2.BORDER_REFLECT,1)
    s_img = cv2.warpAffine(img, M, (dst_size, dst_size))
 

    # for k in range(label.shape[0]): label[k, :] = np.dot(M, np.array( [label[k, 0], label[k, 1], 1]).transpose()).transpose()  

    return s_img,M


cap=cv2.VideoCapture('/Users/momo/Downloads/test_Video/555.mp4')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = 12 if torch.cuda.is_available() else 0
criterion = torch.nn.MSELoss()
net = BigResNet('BigResNet', 3, 120)
net.load_state_dict(torch.load('../checkpoints2/BigResNet10000.pth', map_location='cpu'))
net.to(device)
net.eval()                 


j=0
i=0
points=[]
flag=0

while(cap.isOpened() and j<=1000):
    t0=time.time()   
    ret,frame=cap.read()
    if(ret):
        if flag==0:
            flag=1            
            
            cv2.imwrite('/Users/momo/Downloads/Video1/'+str(j)+'.jpg',frame)
            j=j+1
            img = frame
            frame1=frame.copy()
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',on_mouse)
            cv2.imshow('image',frame)
            kalman=Kalman2D()
            if(cv2.waitKey(0)==27):
                # ROI=img[point1[1]:point2[1],point1[0]:point2[0],:]

                s1_frame,M=get_small_img_1(frame)
                s_frame=s1_frame.copy()
                cv2.imshow('image1',s_frame)
                              
                s_frame=s_frame.transpose((2, 0, 1))
                s_frame=s_frame.reshape(-1,3,128,128)
                inputs = torch.FloatTensor(s_frame)               
                inputs= Variable(inputs)     

                inputs = inputs.to(device)
                outputs = net(inputs)[0]
                outputs=outputs.detach().numpy()
                result=outputs.reshape(2,60)*128
                result_pre=result.copy()

                s=result.copy()
                

                down=M[1,0]*M[0,1]-M[0,0]*M[1,1]            
                
                dup=[ [-M[1,1] / down , M[1,0] / down],[ M[0,1] /down , -M[0,0]/down] ]
                plus=[(M[0,2] * M[1,1]- M[0,1] * M[1,2]) / down, (M[0,0] * M[1,2] - M[1,0] * M[0,2]) / down ]                
                plus=np.array(plus)
                result=np.array(result).T               
                
                prelabel=np.dot(result,np.array(dup))+plus
                prelabel=np.rint(prelabel).astype(np.int32)
                prelabel=prelabel+np.array(point1)
                
                p0=prelabel.astype(np.float32)             


            # for j in range(result.shape[1]):
            #         cv2.circle(s1_frame, (result[0, j], result[1, j]), 2, (0, 255, 0), -1)
                for k in range(result.shape[0]):
                        cv2.circle(frame, (prelabel[k,0], prelabel[k,1]), 3, (0, 255, 0), -1)
                h=int(frame.shape[0]*0.5)
                w=int(frame.shape[1]*0.5)

                frame_show=cv2.resize(frame,(w,h))
                cv2.imshow('image',frame_show)
                cv2.waitKey(0)


        else:
            t0=time.time()
            #ROI=frame[point1[1]:point2[1],point1[0]:point2[0],:]
            frame2=frame.copy()
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            
            p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p0, None, **lk_params)
            state=checkedTrace(frame1_gray,frame2_gray,p0)
            for i in range(len(state)):
                if state[i]==False:
                    p1=p0
            frame1=frame2

            s1_frame,M=get_small_img_3(frame,p1)

            #p0=p_groudT 

            s_frame=s1_frame.copy()
            cv2.imshow('image1',s_frame)
           
                    
            s_frame=s_frame.transpose((2, 0, 1))
            s_frame=s_frame.reshape(-1,3,128,128)
                            

            
            inputs = torch.FloatTensor(s_frame)               
            inputs = Variable(inputs)     

            inputs = inputs.to(device)
            outputs = net(inputs)[0]
            outputs=outputs.detach().numpy()
            result=outputs.reshape(2,60)
            # print '12:',result[0,12],result[1,12]
            # print '44:',result[0,44],result[1,44]
            
            result=result*128 
            for n in range(44,result.shape[1]):                
                result[0,n]=result_pre[0,n]+fdis(result[0,n]-result_pre[0,n])
                result[1,n]=result_pre[1,n]+fdis(result[1,n]-result_pre[1,n])

            dis_s=np.sqrt((result[0,12]-result_pre[0,12])* (result[0,12]-result_pre[0,12])+(result[1,12]-result_pre[1,12])* (result[1,12]-result_pre[1,12]))
            dis_s1=np.sqrt((result[0,44]-result_pre[0,44])* (result[0,44]-result_pre[0,44])+(result[1,44]-result_pre[1,44])* (result[1,44]-result_pre[1,44]))

            # file_p0.write(str(j)+' '+str(dis_s)+'\n')
            file_kalman.write(str(j)+' '+str(dis_s1)+'\n')
            result_pre=result.copy()

            s_label=result.copy()              

            down=M[1,0]*M[0,1]-M[0,0]*M[1,1]            
            
            dup=[ [-M[1,1] / down , M[1,0] / down],[ M[0,1] /down , -M[0,0]/down] ]
            plus=[(M[0,2] * M[1,1]- M[0,1] * M[1,2]) / down, (M[0,0] * M[1,2] - M[1,0] * M[0,2]) / down ]                
            plus=np.array(plus)
            result=np.array(result).T               
            
            prelabel=np.dot(result,np.array(dup))+plus
            prelabel=np.rint(prelabel).astype(np.int32)
            prelabel=prelabel

            # saveLog(prelabel,file_p0) 
            # p0=prelabel.astype(np.float32)    
            #-------------------------------------------- 
            # p_kal=prelabel.copy()
            # # print 'p_kal:',p_kal
            # p_kal=np.array(p_kal)
            # kalman.update(p_kal)
            # est=kalman.getEstimate()
            # est=np.rint(est).astype(np.int16)  
            # p0=est
            # saveLog(est.astype(np.int16),file_kalman) 
            #--------------------------------------------
            # dis=prelabel-p0
            
            # dis_plus=abs(dis)
            # dis_sigmoid=sigmoid(dis_plus)
            # dis=np.multiply(dis_sigmoid,dis)
            
            # prelabel=p0+dis
            p0=updatePoint(p0,prelabel)
            p0=prelabel.astype(np.float32)
            # saveLog(p0.astype(np.int16),file_smooth)
            #-------------------------------------------- 
            
                    


            # for j in range(result.shape[1]):
            #         cv2.circle(s1_frame, (result[0, j], result[1, j]), 2, (0, 255, 0), -1)
            for k in range(result.shape[0]):
                    cv2.circle(frame, (p0[k,0], p0[k,1]), 2, (0, 0, 255), -1)
            
            frame_show=cv2.resize(frame,(w,h))
            cv2.imshow('image',frame_show)
            j=j+1
            t1=time.time()
            # print t1-t0
            if cv2.waitKey(1)==27:
                print j
                break            
            

