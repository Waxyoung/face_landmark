import sys
import os
#from resnet_model import *
import torch.optim as optim
import torch
import numpy as np
import math
import cv2
# from SmallConvNet  import SmallConvNet
from BigResNet import BigResNet 
from my_face_dataset import FaceDataSet
     
if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    num_workers = 12 if torch.cuda.is_available() else 0

    testset = FaceDataSet('../testData')
   
    testloader = torch.utils.data.DataLoader(testset, batch_size = 36, shuffle = True, num_workers = num_workers, drop_last = True)
   
    criterion = torch.nn.MSELoss()

    net = BigResNet('BigResNet', 3, 120)    
    #net = SmallConvNet('SmallConvNet', 3, 18)
    #epoch_list = [50 * (i+1) for i in range(21, 62)]

    # for epoch in epoch_list:
    #     net = torch.load('../checkpoints/backupmodel%05d.pt'%epoch, map_location='cpu')
    #     net.to(device)
    #     net.eval()
    net.load_state_dict(torch.load('../checkpoints6/BigResNet04800.pth', map_location='cpu'))
    # print(net)
    net.to(device)
    net.eval()
    F=0
    fengbu={'0-1':0,'1-2':0,'2-3':0,'3-4':0,'4-5':0,'5-6':0,'6-7':0,'7-8':0,'8-':0}
 
    with torch.no_grad():
        running_loss = 0.0
        cnt = 0
        pix_list=[]
        for i, data in enumerate(testloader):
            inputs, labels = data['img'], data['label']
            
   
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = net(inputs)[0]
  
            loss = criterion(outputs, labels)
            
            #running_loss += loss.item()
            cnt += 1

            batch_size =36
            dst_size = 128

            inputs = inputs.numpy()
            labels = labels.numpy() 
            outputs = outputs.numpy()        
            

            show_length = int(math.sqrt(batch_size))
            show_sample_img = np.zeros((int(show_length * dst_size), int(show_length * dst_size), 3))
            for m in range(show_length):
                for n in range(show_length):
                    img = inputs[m * show_length + n, :, :, :]
                    img = img.transpose((1, 2, 0)).astype(np.uint8)
                    img_clone = np.zeros(img.shape, dtype=np.uint8)
                    img_clone[...] = img
                    label = np.reshape(outputs[m * show_length + n, :], 
                                        (2, outputs.shape[1] / 2)) * dst_size
                    gt = np.reshape(labels[m * show_length + n, :], 
                                        (2, outputs.shape[1] / 2)) * dst_size
                    label = label.astype(np.int32)
                    gt=gt.astype(np.int32)
                    # print '-------'+str(i)+'----------'                  
                    pix=(gt-label)**2
                    pix=np.sqrt(pix.sum(axis=0))
                    pix=pix.sum().astype(np.int32)
                    pix=pix/60.
                    
                    if(0.<pix and pix <=1.): 
                        fengbu['0-1']+=1
                    if(1.<pix and pix <=2.): 
                        fengbu['1-2']+=1
                    if(2.<pix and pix <=3.): 
                        fengbu['2-3']+=1
                    if(3.<pix and pix <=4.):   
                        fengbu['3-4']+=1
                    if(4.<pix and pix <=5.):      
                        fengbu['4-5']+=1
                    if(5.<pix and pix <=6.):
                        fengbu['5-6']+=1
                    if(6.<pix and pix <=7.):
                        fengbu['6-7']+=1
                    if(7.<pix and pix <=8.): 
                        fengbu['7-8']+=1
                    if(8.<pix): 
                        fengbu['8-']+=1
                        for j in range(label.shape[1]):
                            cv2.circle(img_clone, (label[0, j], label[1, j]), 2, (0, 255, 0), -1)
                        cv2.putText(img_clone,str(int(pix)),(5,20),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,255),1)     
                        cv2.imwrite('/Users/momo/Downloads/false/2350-no_dataAug/'+str(fengbu['8-'])+'-'+str(int(pix))+'.jpg',img_clone)
                    pix_list.append(pix)
                    # print 'pix',pix      
                    # print label
                    # print '-------'+str(i)+'----------'
                    cv2.putText(img_clone,str(int(pix)),(5,20),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,255),1)
                    for j in range(label.shape[1]):
                            cv2.circle(img_clone, (label[0, j], label[1, j]), 1, (0, 255, 0), -1)
                    show_sample_img[n * dst_size:(n + 1) * dst_size, 
                    m * dst_size: (m + 1) * dst_size] = img_clone
            show_sample_img = show_sample_img.astype(np.uint8)
            cv2.imshow('show_sample_img1', show_sample_img)
            # cv2.imwrite('/Users/momo/Downloads/imageScan/noShelter_sizeSmall_moreImg'+str(i)+'.jpg',show_sample_img)
            if(cv2.waitKey()==27):
                break
        
        pix_list=np.array(pix_list)
        pix_list=pix_list
        print 'fengbu:',fengbu   
        print 'pix-shape:',pix_list.shape
        print 'pix-mean:',pix_list.mean()
        print 'pix-var:',pix_list.var()
            # show_sample_img = np.zeros((int(show_length * dst_size), int(show_length * dst_size), 3))
            # for m in range(show_length):
            #     for n in range(show_length):
            #         img = inputs[m * show_length + n, :, :, :]
            #         img = img.transpose((1, 2, 0)).astype(np.uint8)
            #         img_clone = np.zeros(img.shape, dtype=np.uint8)
            #         img_clone[...] = img
            #         label = np.reshape(labels[m * show_length + n, :],
            #                             (2, outputs.shape[1] / 2)) * dst_size
            #         label = label.astype(np.int32)
            #         for j in range(label.shape[1]):
            #                 cv2.circle(img_clone, (label[0, j], label[1, j]), 1, (255, 0, 0), 1)
            #         show_sample_img[n * dst_size:(n + 1) * dst_size,
            #         m * dst_size: (m + 1) * dst_size] = img_clone
            # show_sample_img = show_sample_img.astype(np.uint8)
            # cv2.imshow('show_sample_img', show_sample_img)
            # cv2.waitKey()


        #print 'epoch: %d RMSE Loss:%.5f' % ( epoch, math.sqrt(running_loss / cnt) )

            


