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
def getImage_wellClassified(num,imshow,dataset):
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
        if imshow==True:
            for i,img in enumerate(correctImgs):
                plt.subplot(math.ceil(len(correctImgs)/4),4,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(img)
                plt.xlabel(correctLabelName[i],fontsize=5)
            plt.show() 
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


def fgsm_attack(model, loss_fn, x, y, epsilon,norm):
    # 创建对x的梯度记录
    x = tf.Variable(x, dtype=tf.float32, trainable=True)
    with tf.GradientTape() as tape:
        y_pred = model(x, training=False)
        loss = loss_fn(y, y_pred)
    # 获取关于图像的梯度
    gradient = tape.gradient(loss, x)
    if norm=="inf":
        # 计算符号梯度
        signed_grad = tf.sign(gradient)
        # 添加扰动
        x_adv = tf.add(x, epsilon*signed_grad)
    elif norm=="2":
        # 计算符号梯度并对其进行归一化处理
        norm_grad = tf.linalg.normalize(gradient)[0]
        # 添加扰动
        x_adv = tf.add(x, epsilon * norm_grad)
    elif norm=="1":
        # 计算符号梯度并对其进行归一化处理
        norm_grad = tf.math.sign(gradient)
        abs_grad = tf.math.abs(norm_grad)
        norm = tf.math.reduce_sum(abs_grad)
        signed_grad = epsilon * norm_grad / norm
        x_adv = tf.add(x, epsilon * signed_grad)
    # 返回扰动后的输入样本
    return np.array(x_adv)


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

def createAdversarialSamples(sampleNum,epsilon,dataset,norm):
    #定义模型，损失函数
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    if dataset=="ImageNet":
        #获取模型分类正确的图像数据
        labelName,imgs,imgData,predictions,count=getImage_wellClassified(sampleNum,False,"ImageNet")
        model=tf.keras.applications.resnet50.ResNet50(weights='imagenet')
        max_index = np.argmax(predictions, axis=1)
        #获得真实标签
        ground_truth = np.zeros_like(predictions)
        ground_truth[np.arange(len(predictions)), max_index] = 1
        #产生对应的对抗样本
        img_adv_data = [fgsm_attack(model=model, loss_fn=loss_fn, x=np.expand_dims(imgData[i], axis=0), y=np.expand_dims(ground_truth[i], axis=0), epsilon=epsilon,norm=norm) for i in range(len(ground_truth))]
        img_adv_arr = [np.squeeze(img_adv_data[i]) for i in range(len(img_adv_data))]
        img_adv_arr = [resnet50_deprocess(img_adv_arr[i]) for i in range(len(img_adv_arr))]
        img_advs = [PIL.Image.fromarray(img_adv_arr[i].astype('uint8'))for i in range(len(img_adv_arr))]
        num=0
        for img_raw,img_adv in zip(imgs,img_advs):
            flag,label_raw,label_adv=isAdversarial_ResNet50(img_raw,img_adv)
            if flag:
                raw_path="./RAW/fgsm_raw/imageNet_"+str(label_raw)+"_"+str(num+1)+"_"+str(epsilon)+"_"+str(norm)+".jpeg"
                adv_path="./Adversarial_Samples/fgsm/imageNet_"+str(label_adv)+"_"+str(num+1)+"_"+str(epsilon)+"_"+str(norm)+".jpeg"
                img_raw.save(raw_path)
                img_adv.save(adv_path)
                num+=1 
        print("生成对抗样本数量：",num)         
        print("生成对抗样本成功率为：",num/count)
    elif dataset=="Cifar10":
        #获取模型分类正确的图像数据
        labelName,imgs,imgData,predictions,count=getImage_wellClassified(sampleNum,False,"Cifar10")
        model=load_model("D:/Adversarial Attack/testModel/cifar10_ResNet20.hdf5")
        #获得真实标签
        max_index = np.argmax(predictions, axis=1)
        ground_truth = np.zeros_like(predictions)
        ground_truth[np.arange(len(predictions)), max_index] = 1
        #产生对应的对抗样本
        img_adv_data = [fgsm_attack(model=model, loss_fn=loss_fn, x=np.expand_dims(imgData[i], axis=0), y=np.expand_dims(ground_truth[i], axis=0), epsilon=epsilon,norm=norm) for i in range(len(imgData))]
        img_adv_arr = [np.squeeze(img_adv_data[i]) for i in range(len(img_adv_data))]
        img_adv_arr = [(img_adv_arr[i]*255) for i in range(len(img_adv_arr))]
        img_advs = [PIL.Image.fromarray(img_adv_arr[i].astype('uint8'))for i in range(len(img_adv_arr))]
        num=0
        for img_raw,img_adv in zip(imgs,img_advs):
            flag,label_raw,label_adv=isAdversarial_ResNet20(img_raw,img_adv)
            if flag:
                raw_path="./RAW/fgsm_raw/cifar10_"+str(label_raw)+"_"+str(num+1)+"_"+str(epsilon)+"_"+str(norm)+".jpeg"
                adv_path="./Adversarial_Samples/fgsm/cifar10_"+str(label_adv)+"_"+str(num+1)+"_"+str(epsilon)+"_"+str(norm)+".jpeg"
                img_raw.save(raw_path)
                img_adv.save(adv_path)
                num+=1 
        print("生成对抗样本数量：",num)         
        print("生成对抗样本成功率为：",num/count)


if __name__ =='__main__':
    # createAdversarialSamples(100,0.05,"Cifar10","2")
    # createAdversarialSamples(100,0.3,"ImageNet","2")
    # createAdversarialSamples(100,0.05,"Cifar10","inf")
    # createAdversarialSamples(100,0.3,"ImageNet","inf")
    createAdversarialSamples(100,0.05,"Cifar10","1")
    createAdversarialSamples(100,0.3,"ImageNet","1")
    
    
    

    