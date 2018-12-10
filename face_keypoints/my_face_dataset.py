import os
import sys
import re
import math
import cv2
import numpy as np
import collections 
import torch.utils.data as data
import random
import torch

left_nose_idx=12
right_nose_idx=20

DataPath = collections.namedtuple('DataPath', ['imgfile_fullpath', 'ptfile_fullpath'])
ImgWithLabel = collections.namedtuple('ImgWithLabel', ['img', 'label'])

dst_dir='/Users/momo/Downloads//60point/test_Data/'
landmark_order=[6,8,7,10,9,1,3,2,5,4,
                20,22,21,23,24,26,25,28,27,
                11,13,12,14,15,17,16,19,18,
                30,29,31,32,
                33,35,34,39,38,37,36,40,44,43,42,41,
                45,46,48,47,50,49,52,51,60,59,58,57,56,55,54,53]
class FaceDataSet(data.Dataset):
    def __init__(self, root, load_size =300):
        super(FaceDataSet, self).__init__()        
        self.all_data = []
        self.dst_size = 128
        self.src_map = {}

        pattern = re.compile(r'^\w.+\.jpg$')
        for dirpath, dirnames, filenames in os.walk(root):
            for img_filename in filenames:

                match = pattern.match(img_filename)
                if not match: continue
                
                pt_filename = img_filename.replace('.jpg', '.pt')
                ptfile_fullpath = os.path.join(dirpath, pt_filename)
                imgfile_fullpath = os.path.join(dirpath, img_filename)
                if os.path.exists(ptfile_fullpath) and os.path.exists(imgfile_fullpath):
                    self.all_data.append(DataPath(imgfile_fullpath = imgfile_fullpath, ptfile_fullpath = ptfile_fullpath))

        print('All data length: %d' % len(self.all_data))

        for index in range(len(self.all_data)):
            if index % 1000 == 0: 

                print('load data %d'%index)                
                sys.stdout.flush()
                
            img = cv2.imread(self.all_data[index].imgfile_fullpath)
            label = np.loadtxt(self.all_data[index].ptfile_fullpath, skiprows=1).astype(np.float32)
        # print(self.all_data[1].ptfile_fullpath)
        # label = np.loadtxt(self.all_data[1].ptfile_fullpath, skiprows=1).astype(np.float32)
        # print(label[1,:])

            self.src_map[self.all_data[index].imgfile_fullpath] = self.__get_small_img(img, label, dst_size = load_size, face_scale = 0.6)

        
    def __get_small_img(self, img, label, dst_size = 300, face_scale = 0.9, data_aug = False):
        if data_aug and (random.random() < 0.2):
            rotate_degree = (random.random() - 0.5) * 2 * 20
            M_tmp = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), rotate_degree, 1.0)

            tmp_factor = random.random()
            if tmp_factor < 0.4:
                offset = np.array([(random.random() - 0.5) * 2.0 * img.shape[0]/2, 
                                (random.random() - 0.5) * 1.0 * img.shape[0]/2 ])
            elif tmp_factor < 0.7:
                offset = np.array([(random.random() - 0.5) * 1.0 * img.shape[0]/2, 
                                (random.random() - 0.5) * 2.0 * img.shape[0]/2 ])
            else:
                offset = np.array([(random.random() - 0.5) * 2.0 * img.shape[0]/2, 
                                (random.random() - 0.5) * 2.0 * img.shape[0]/2 ])

            M_tmp[:, 2] += offset
            img = cv2.warpAffine(img, M_tmp, (img.shape[0], img.shape[1]))
            M_tmp = cv2.getRotationMatrix2D((img.shape[0]/2 + offset[0], img.shape[1]/2 + offset[1]), -rotate_degree, 1.0)
            M_tmp[:, 2] -= offset
            img = cv2.warpAffine(img, M_tmp, (img.shape[0], img.shape[1]))

        if data_aug and (random.random() < 0.5):
            img = cv2.flip(img, 1)
            swap_label = np.copy(label)
            swap_label[:, 0] = img.shape[1] - 1 - swap_label[:, 0]
            landmark_order_num = len(landmark_order)
            for label_i in range(landmark_order_num):
                label[label_i, :] = swap_label[landmark_order[label_i]-1, :]
            # landmarks_order_num = len(landmarks_order_107)
            # for label_i in range(landmarks_order_num):
            #     label[label_i, :] = swap_label[landmarks_order_107[label_i], :]
            
        
        src_left_nose = label[left_nose_idx, :]
        src_right_nose = label[right_nose_idx, :]
        src_middle = (src_left_nose + src_right_nose) / 2
        # src_middle = (src_middle + label[2,:]) / 2

        top_mid = (label[4,:]+label[7,:]) /2

        # disV = cv2.norm(top_mid  - label[2])
        # disH = cv2.norm(label[4]  - label[7])
        # face_size = max(disV*0.95, disH*0.95)

        
        # src_length = min(max(cv2.norm(src_left_nose - src_right_nose), face_size * 0.15), face_size * 0.3)
        src_length = cv2.norm(src_left_nose,src_right_nose)
        dst_middle = np.array([0.5, 0.58]) * dst_size #300:150x174
        dst_length = dst_size * 0.16  #300:54

        if data_aug:
            if random.random() < 0.1:
                dst_middle += np.array(
                            [(random.random() - 0.5) * 2 * 0.3 * dst_size,
                            (random.random() - 0.5) * 2 * 0.3 * dst_size])
            elif random.random() < 0.5:
                dst_middle += np.array(
                            [(random.random() - 0.5) * 2 * 0.15 * dst_size,
                            (random.random() - 0.5) * 2 * 0.15 * dst_size])
            else:
                dst_middle += np.array(
                            [(random.random() - 0.5) * 2 * 0.05 * dst_size,
                            (random.random() - 0.5) * 2 * 0.05 * dst_size])
        
        if data_aug: 
            if random.random() < 0.7:
                face_scale += (random.random()-0.5) * 2 * 0.4
            else:
                face_scale += (random.random()-0.5) * 2 * 0.2
        scale = dst_length / src_length * face_scale
        offset = dst_middle - src_middle

        rotate_degree = 0
        rotate_radian = math.atan2( (label[right_nose_idx][1] - label[left_nose_idx][1]) , (label[right_nose_idx][0] - label[left_nose_idx][0]) )
        rotate_degree = rotate_radian / 3.1415926 * 180
        if data_aug: 
            if random.random() < 0.7:
                rotate_degree += (random.random() - 0.5) * 2 * 40
            else:
                rotate_degree += (random.random() - 0.5) * 2 * 6

        M = cv2.getRotationMatrix2D((src_middle[0], src_middle[1]), rotate_degree, scale)
        M[:, 2] += offset
        #img = cv2.warpAffine(img, M, (int(np.math.ceil(128.0)), int(np.math.ceil(128.0))),cv2.INTER_LINEAR,cv2.BORDER_REFLECT,1)
        img = cv2.warpAffine(img, M, (dst_size, dst_size))
        if data_aug and  random.randint(0, 1) == 1:
            max_size = int(min(img.shape[0], img.shape[1]) * 0.05)
            if max_size == 0:
                max_size = 3
            kernel_size = np.random.randint(max_size) 
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        for j in range(label.shape[0]): label[j, :] = np.dot(M, np.array( [label[j, 0], label[j, 1], 1]).transpose()).transpose()
        
        # for i in range(label.shape[0]):
        #     cv2.circle(img, (label[i, 0], label[i, 1]), 1, (255, 0, 0), 1)
        
        
        # cv2.imshow('img', img)

        # cv2.waitKey()
        

        return ImgWithLabel(img = img, label = label)

    
    def __getitem__(self, index):
            img = self.src_map[self.all_data[index].imgfile_fullpath].img.copy()
            label = self.src_map[self.all_data[index].imgfile_fullpath].label.copy()

            face_set = self.__get_small_img(img, label, dst_size = 128, face_scale = 0.95, data_aug =True)
            
            img = face_set.img.transpose((2, 0, 1))
           
            label = label.transpose().reshape(-1) / self.dst_size 

            return {'img':torch.FloatTensor(img), 'label':torch.FloatTensor(label)}


    def __len__(self):
        return len(self.all_data)

if __name__=='__main__':
    trainset = FaceDataSet('/Users/momo/Downloads/60point/testData')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=False, num_workers = 0, drop_last = True)
 
    for epoch in range(1):
        print('epoch %d' % epoch)
        sys.stdout.flush()
        for i, batch in enumerate(trainloader):

            batch['img'] = batch['img'].numpy()
            batch['label'] = batch['label'].numpy()

            batch_size = 36
            dst_size = 128

            show_length = int(math.sqrt(batch_size))
            show_sample_img = np.zeros((int(show_length * dst_size), int(show_length * dst_size), 3))
            for m in range(show_length):
                for n in range(show_length):
                    img = batch['img'][m * show_length + n, :, :, :] 
                    img = img.transpose((1, 2, 0)).astype(np.uint8)
                    img_clone = np.zeros(img.shape, dtype=np.uint8)
                    img_clone[...] = img
                    label = np.reshape(batch['label'][m * show_length + n, :],
                                        (2, int(batch['label'].shape[1] / 2))) * dst_size
                    label = label.astype(np.int32)
                    for j in range(label.shape[1]):
                            cv2.circle(img_clone, (label[0, j], label[1, j]), 1, (0, 0, 255), -1)
                    show_sample_img[n * dst_size:(n + 1) * dst_size,
                    m * dst_size: (m + 1) * dst_size] = img_clone
        
            show_sample_img = show_sample_img.astype(np.uint8)
            cv2.imshow('show_sample_img', show_sample_img)
            #cv2.imwrite('/Users/momo/Downloads/imageScan/Gussain_rotate_scale_size(1)'+str(i)+'.jpg',show_sample_img)
           
            if(cv2.waitKey()==27):                 
                exit()
