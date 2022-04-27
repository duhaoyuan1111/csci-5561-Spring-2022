import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC
from scipy import stats
from pathlib import Path, PureWindowsPath
import math


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def get_tiny_image(img, output_size):
    # 16*16
    h,w = np.shape(img)
    wout = output_size[0]
    hout = output_size[1]
    feature = np.zeros([hout,wout])
    blockh = math.ceil(h/hout)
    blockw = math.ceil(w/wout)
    for i in range(hout):
        for j in range(wout):
            feature[i,j] = np.sum(img[i*blockh:(i+1)*blockh, j*blockw:(j+1)*blockw]) / (blockh*blockw)
    #normalize
    mean = np.mean(feature)
    norm = np.linalg.norm(feature,2)
    feature = (feature - mean)/norm
    
    return feature.reshape(wout*hout)


def predict_knn(feature_train, label_train, feature_test, k):
    # label_train [0,14]
    neigh = NearestNeighbors().fit(feature_train)
    #neigh
    dist,ind = neigh.kneighbors(feature_test,n_neighbors=k)
    label_test_pred = np.zeros(len(dist))
    for i in range(len(dist)):
        temp = np.zeros(15)
        for j in range(len(ind[i])):
            if label_train[ind[i,j]] == 0:
                temp[0] += 1
            elif label_train[ind[i,j]] == 1:
                temp[1] += 1
            elif label_train[ind[i,j]] == 2:
                temp[2] += 1
            elif label_train[ind[i,j]] == 3:
                temp[3] += 1
            elif label_train[ind[i,j]] == 4:
                temp[4] += 1
            elif label_train[ind[i,j]] == 5:
                temp[5] += 1
            elif label_train[ind[i,j]] == 6:
                temp[6] += 1
            elif label_train[ind[i,j]] == 7:
                temp[7] += 1
            elif label_train[ind[i,j]] == 8:
                temp[8] += 1
            elif label_train[ind[i,j]] == 9:
                temp[9] += 1
            elif label_train[ind[i,j]] == 10:
                temp[10] += 1
            elif label_train[ind[i,j]] == 11:
                temp[11] += 1
            elif label_train[ind[i,j]] == 12:
                temp[12] += 1
            elif label_train[ind[i,j]] == 13:
                temp[13] += 1
            elif label_train[ind[i,j]] == 14:
                temp[14] += 1
        #print(temp)
        result = np.argmax(temp)
        #print(result)
        label_test_pred[i] = result

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    
    k = 9 # [7,0.216] [8,0.217] [9,0.218] [10,0.217] [11,0.215]

    # tiny the train & test
    trainArray = []
    testArray = []
    labelTrainArray = []
    labelTestArray = []
    for i in range(len(img_train_list)):
        eachImg = cv2.imread(img_train_list[i],0)
        tinyTrain = get_tiny_image(eachImg,[16,16])
        trainArray.append(tinyTrain)
    for j in range(len(img_test_list)):
        eachImg = cv2.imread(img_test_list[j],0)
        tinyTest = get_tiny_image(eachImg,[16,16])
        testArray.append(tinyTest)
    for m in range(len(label_train_list)):
        labelTrainArray.append(label_classes.index(label_train_list[m]))
    for n in range(len(label_test_list)):
        labelTestArray.append(label_classes.index(label_test_list[n]))
    
    trainArray = np.array(trainArray)
    testArray = np.array(testArray)
    labelTrainArray = np.array(labelTrainArray)
    labelTestArray = np.array(labelTestArray)
    
    predict_result = predict_knn(trainArray, labelTrainArray, testArray, k)
    #print(predict_result) # float, should be int
    #print("\n\n")
    #print(labelTestArray)
    
    #Accuracy (all correct / all) = TP + TN / TP + TN + FP + FN
    #https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    confusion = np.zeros([15,15])
    for e in range(len(predict_result)):
        if predict_result[e] == labelTestArray[e]:
            confusion[int(predict_result[e]),int(predict_result[e])] += 1
        else:
            confusion[labelTestArray[e], int(predict_result[e])] += 1
    accuracy = 0
    for p in range(len(confusion)):
        accuracy += confusion[p][p]
    
    accuracy = accuracy / np.sum(confusion)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def compute_dsift(img, stride, size):
    #To do
    sift = cv2.xfeatures2d.SIFT_create()
    # stride: ratio
    # size: length of block
    dense_feature = [] #n*128
    h,w = np.shape(img)
    blockh = math.floor((h-size)/stride) # how many blocks I can get
    blockw = math.floor((w-size)/stride)
    #compute(InputArrayOfArrays images, std::vector<std::vector<KeyPoint>> &keypoints, OutputArrayOfArrays 	descriptors )
    #a = cv2.KeyPoint(2,3,5)
    #print(a.pt)
    for i in range(blockh):
        for j in range(blockw):
            startPointH = i*stride
            startPointW = j*stride
            #block = img[startPointH:startPointH+size , startPointW:startPointW+size]
            keypoint = cv2.KeyPoint(startPointW+(size//2),startPointH+(size//2),size)
            keypoints, descriptors = sift.compute(img, [keypoint])
            #print(descriptors,"   " ,len(descriptors),"   ",np.shape(descriptors)))
            for k in range(len(descriptors)):
                oneDDes = descriptors[k].reshape(-1)
                dense_feature.append(oneDDes)

    dense_feature = np.array(dense_feature)
    return dense_feature #n*128


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    print("Go to a snap")
    dic_size = 220

    denseKMeans = np.empty((1,128))
    for i in range(len(dense_feature_list)):
       denseKMeans = np.concatenate((denseKMeans, dense_feature_list[i]))

    km = KMeans(max_iter=200, n_init=10, n_clusters=dic_size).fit(denseKMeans[1:,])
    vocab = km.cluster_centers_
    print("Vocab Done")
    return vocab # dic_size¡Á128 = 50*128


def compute_bow(feature, vocab):
    dic_size = len(vocab) # 50
    neigh = NearestNeighbors().fit(vocab)
    #neigh
    dist,ind = neigh.kneighbors(feature,n_neighbors=1)
    ind = ind.reshape(-1)
    bow_feature = np.zeros(dic_size) # 50
    for i in range(len(ind)):
        bow_feature[ind[i]] += 1
    norm = np.linalg.norm(bow_feature)
    bow_feature = bow_feature/norm
    
    return bow_feature # 1*50


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # [18,18] k=9 iter 260 dic 50 (0.501)
    # [26,26] k=9 iter 200 dic 220 (0.517)
    dic_size = 220
    stride = 26
    size = 26
    denseArray = []
    bowArray = []
    for i in range(len(img_train_list)):
        tempimg = cv2.imread(img_train_list[i],0)
        #print("1111111111")
        dense = compute_dsift(tempimg, stride, size)
        #print("2222222222")
        denseArray.append(dense)
    
    denseArray = np.array(denseArray)
    visual_dic = build_visual_dictionary(denseArray, dic_size)
    for j in range(len(denseArray)):
        bow = compute_bow(denseArray[j], visual_dic)
        bowArray.append(bow)
    bowArray = np.array(bowArray)
    #print("33333333333")
    bowTestArray = []
    for k in range(len(img_test_list)):
        tempimg = cv2.imread(img_test_list[k],0)
        dense = compute_dsift(tempimg, stride, size)
        bowTest = compute_bow(dense, visual_dic)
        bowTestArray.append(bowTest)
    bowTestArray = np.array(bowTestArray)

    labelTrainArray = []
    labelTestArray = []
    for m in range(len(label_train_list)):
        labelTrainArray.append(label_classes.index(label_train_list[m]))
    for n in range(len(label_test_list)):
        labelTestArray.append(label_classes.index(label_test_list[n]))
    labelTrainArray = np.array(labelTrainArray)
    labelTestArray = np.array(labelTestArray)

    k = 9

    predict_result = predict_knn(bowArray, labelTrainArray, bowTestArray, k)
    confusion = np.zeros([15,15])
    for e in range(len(predict_result)):
        if predict_result[e] == labelTestArray[e]:
            confusion[int(predict_result[e]),int(predict_result[e])] += 1
        else:
            confusion[labelTestArray[e], int(predict_result[e])] += 1
    accuracy = 0
    for p in range(len(confusion)):
        accuracy += confusion[p][p]
    
    accuracy = accuracy / np.sum(confusion)


    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    
    label_test_pred = []
    for i in range(15):
        svc = LinearSVC(C = 0.5)
        biLabel = np.zeros(len(label_train))
        for j in range(len(label_train)):
            if label_train[j] == i:
                biLabel[j] = 1
            else:
                biLabel[j] = 0
        svc.fit(feature_train, biLabel)
        result = svc.decision_function(feature_test)
        # print(len(result)) # 1D
        result = result.reshape(-1)
        
        label_test_pred.append(result) # 15*1500

    label_test_pred = np.array(label_test_pred)
    ans = np.argmax(label_test_pred, axis = 0)
    return ans # 1*1500



def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # [22,22] iter 200 dic 220 C 0.5 time 40:47 (0.611) 36:09 (0.618) 32:04 (0.617)


    dic_size = 220
    stride = 22
    size = 22

    denseArray = []
    bowArray = []
    for i in range(len(img_train_list)):
        tempimg = cv2.imread(img_train_list[i],0)
        #print("1111111111")
        dense = compute_dsift(tempimg, stride, size)
        #print("2222222222")
        denseArray.append(dense)
    denseArray = np.array(denseArray)
    visual_dic = build_visual_dictionary(denseArray, dic_size)
    for j in range(len(denseArray)):
        bow = compute_bow(denseArray[j], visual_dic)
        bowArray.append(bow)
    bowArray = np.array(bowArray)
    #print("33333333333")
    bowTestArray = []
    for k in range(len(img_test_list)):
        tempimg = cv2.imread(img_test_list[k],0)
        dense = compute_dsift(tempimg, stride, size)
        bowTest = compute_bow(dense, visual_dic)
        bowTestArray.append(bowTest)
    bowTestArray = np.array(bowTestArray)
    labelTrainArray = []
    labelTestArray = []
    for m in range(len(label_train_list)):
        labelTrainArray.append(label_classes.index(label_train_list[m]))
    for n in range(len(label_test_list)):
        labelTestArray.append(label_classes.index(label_test_list[n]))
    labelTrainArray = np.array(labelTrainArray)
    labelTestArray = np.array(labelTestArray)

    predict_result = predict_svm(bowArray, labelTrainArray, bowTestArray, 15)
    confusion = np.zeros([15,15])
    for e in range(len(predict_result)):
        if predict_result[e] == labelTestArray[e]:
            confusion[int(predict_result[e]),int(predict_result[e])] += 1
        else:
            confusion[labelTestArray[e], int(predict_result[e])] += 1
    accuracy = 0
    for p in range(len(confusion)):
        accuracy += confusion[p][p]
    
    accuracy = accuracy / np.sum(confusion)


    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path\
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")

    #label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./drive/MyDrive/scene_classification_data")
    # print(len(label_classes))
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    # all: 3/8/2022 23:35 (0.218) (0.511) (0.625) exetime: 53:53


    #!pip3 install opencv-python==3.4.2.17
    #!pip3 install opencv-contrib-python==3.4.2.17
