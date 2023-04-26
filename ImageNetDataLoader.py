from scipy import io
import os
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import numpy as np
class ImageNetData:
    def get_class_name(self,meta, class_id):
        #从ImageNet的meta.mat文件中获取给定类别ID的类别名。
        synsets = meta['synsets']
        # 查找目标类别ID并获取类别名
        target_name = ''
        for i in range(len(synsets)):
            if synsets[i][0][0][0] == class_id:
                target_name = synsets[i][0][1][0]
                break
            
        return target_name
    def loadData(self,num):
        #加载类别信息
        meta = io.loadmat("./ILSVRC2012/ILSVRC2012_devkit_t12/data/meta.mat")
        #加载ImageNetILSVRC2015训练数据集
        with open(file="./ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt") as f:
            ILSVRC2012_val_label=[int(line[:-1])for line in f.readlines()]
        #加载图片列表
        piclist=os.listdir(path="ILSVRC2012/pictures/")
        #加载图片路径列表
        image_paths=[] 
        for pic in piclist:
            image_paths.append("ILSVRC2012/pictures/"+pic)
        
        val_ids=[]
        ILSVRC_IDs=[]
        class_ids=[]
        # 加载和预处理每个图像，并将它们添加到一个NumPy数组中
        imagedata = np.zeros((num, 224, 224, 3), dtype=np.float32)
        i=0
        imgs=[]
        np.random.seed(100)
        random_arr = np.random.choice(50000, num, replace=False)
        # print(len(random_arr))
        # print(random_arr)
        #读取验证id，类别标签和类别名称
        for randint in random_arr:
            val_id = int(piclist[randint].split('.')[0].split('_')[-1])
            val_ids.append(val_id)
            ILSVRC_ID = ILSVRC2012_val_label[val_id-1]
            ILSVRC_IDs.append(ILSVRC_ID)
            class_id = meta['synsets'][ILSVRC_ID-1][0][1][0].item()
            # synsets = meta['synsets']
            # label_all=synsets[0]
            # print(label_all.tolist())
            # index=label_all.tolist().index(class_id)
            # #获得对应类别名
            # label_Name=synsets[0][index][2][0]
            # label_names.append(label_Name)
            class_ids.append(class_id)
            print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, class_id))
             #加载图像数据
            img = tf.keras.preprocessing.image.load_img(image_paths[randint], target_size=(224, 224))
            imgs.append(img)
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.keras.applications.resnet50.preprocess_input(img,data_format="channels_last")
            imagedata[i] = img
            i+=1           
        return val_ids,ILSVRC_IDs,class_ids,imgs,imagedata
    
        

