from ImageNetDataLoader import ImageNetData
from  Cifar10DataLoader import Cifar10Data
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL
from keras.models import load_model
def count_matching_results(list1, list2):
    count = 0
    matching_ids=[]
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            matching_ids.append(i)
            count += 1
    return count,matching_ids
def getImage_wellClassified(num,dataset):
    correctImgs=[]
    correctLabelName=[]
    correctPredictions=[]
    correctImgData=[]
    count=0
    if dataset=="ImageNet":
        #获得正确分类的图像
        imageNetData=ImageNetData()
        val_id,ILSVRC_ID,groundTruths,imgs,imagedata=imageNetData.loadData(num)
        ResNet50_ImageNet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
        #获得预测结果
        predictions = ResNet50_ImageNet.predict(np.array(imagedata))
        #解析预测结果
        decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)
        #获得标签id和类别名
        predids=[]
        predlabelnames=[]
        for i in decoded_predictions:
            predids.append(i[0][0])
            predlabelnames.append(i[0][1])
        count,matching_ids=count_matching_results(predids,groundTruths)
        print("分类准确率:",count/num) 
        print("分类成功图片数量:",count) 
        #找出分类正确的图片
        for id in matching_ids:
            #记录被正确分类的图片的原图，标签名，以及预测概率向量
            correctImgs.append(imgs[id])
            correctLabelName.append(predlabelnames[id])
            correctPredictions.append(predictions[id])
            correctImgData.append(imagedata[id])
    elif dataset=="Cifar10":
        cifar10Data=Cifar10Data()
        imgs,imagedata,groundTruths=cifar10Data.loadData(num)
        ResNet20_Cifar10=load_model("D:/Adversarial Attack/testModel/cifar10_ResNet20.hdf5")
         #获得预测结果
        predictions=ResNet20_Cifar10.predict(np.array(imagedata))
        predlabelnames = np.argmax(ResNet20_Cifar10.predict(np.array(imagedata)),axis=1).tolist()
        #print(predlabelnames[0])
        count,matching_ids=count_matching_results(predlabelnames,groundTruths)
        print("分类准确率:",count/num) 
        print("分类成功图片数量:",count) 
        for id in matching_ids:
            #记录被正确分类的图片的原图，标签名，以及预测概率向量
            correctImgs.append(imgs[id])
            correctLabelName.append(predlabelnames[id])
            correctPredictions.append(predictions[id])
            correctImgData.append(imagedata[id])
    return correctLabelName,correctImgs,correctImgData,correctPredictions,count  

def resnet50_deprocess(img):
    img /= 1.0
    img += np.array([103.939, 116.779, 123.68])
    img= img[:, :, ::-1]  
    return img

def isAdversarial_ResNet50(img_raw,img_adv):
    #预处理图像
    img_raw = tf.keras.preprocessing.image.img_to_array(img_raw)
    img_raw= tf.keras.applications.resnet50.preprocess_input(img_raw)
    #print(img_raw.shape)

    img_adv=tf.keras.preprocessing.image.img_to_array(img_adv)
    img_adv=tf.keras.applications.resnet50.preprocess_input(img_adv)
    #print(img_adv.shape)
    #加载模型
    model=tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    prediction_raw = model.predict(tf.expand_dims(img_raw, axis=0))
    prediction_adv = model.predict(tf.expand_dims(img_adv, axis=0))
    #比较是否标签
    top_label_raw = tf.keras.applications.imagenet_utils.decode_predictions(prediction_raw, top=1)[0][0][0]
    print("原样本标签：",top_label_raw)
    top_label_adv = tf.keras.applications.imagenet_utils.decode_predictions(prediction_adv, top=1)[0][0][0]
    print("对抗样本标签：",top_label_adv)
    if top_label_raw!=top_label_adv:
        return True,top_label_raw,top_label_adv
    else :
        return False,top_label_raw,top_label_adv

def isAdversarial_ResNet20(img_raw,img_adv):
    #预处理图像
    img_raw = tf.keras.preprocessing.image.img_to_array(img_raw)
    img_raw= img_raw.astype('float32')/255
    #print(img_raw.shape)
    img_adv = tf.keras.preprocessing.image.img_to_array(img_adv)
    img_adv=img_adv.astype('float32')/255
    #print(img_adv.shape)
    #加载模型
    model=load_model("D:/Adversarial Attack/testModel/cifar10_ResNet20.hdf5")
    prediction_raw = model.predict(tf.expand_dims(img_raw, axis=0))
    prediction_adv = model.predict(tf.expand_dims(img_adv, axis=0))
    #比较是否标签
    top_label_raw =np.argmax(np.array(prediction_raw),axis=1)
    print("原样本标签：",top_label_raw)
    top_label_adv = np.argmax(np.array(prediction_adv),axis=1)
    print("对抗样本标签：",top_label_adv)
    if top_label_raw!=top_label_adv:
        return True,top_label_raw,top_label_adv
    else :
        return False,top_label_raw,top_label_adv


    
    
    

    