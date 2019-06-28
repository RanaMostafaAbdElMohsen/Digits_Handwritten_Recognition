import feature_extraction as fe
import glob
import idx2numpy
import numpy as np

def prepare_train():
    X_train = []
    Y_train = []
    path = "Data/MAHDBase_TrainingSet"
    for i in range(12):
        if i<10:
            bmpFiles = glob.glob(path+"/Part0" + str(i) + "/*.bmp")
        else:
            bmpFiles = glob.glob(path+"/Part"+str(i)+"/*.bmp")
        for bmpFile in bmpFiles:
            X_train.append(fe.ReadImage(bmpFile))
            Y_train.append(fe.ExtractClass(bmpFile))

    return X_train,Y_train

def prepare_english_train():
    X_train = []
    Y_train = []

    train_path = "Data/train-images-idx3-ubyte/train-images.idx3-ubyte"
    label_path = "Data/train-labels-idx1-ubyte/train-labels.idx1-ubyte"

    Y_train = idx2numpy.convert_from_file(label_path)
    X_train = np.reshape((idx2numpy.convert_from_file(train_path))/255,(len(Y_train),-1))

    return X_train,Y_train

def prepare_test():
    X_test = []
    Y_test = []
    path = "Data/MAHDBase_TestingSet/"
    for i in range(1,3):
        bmpFiles = glob.glob(path+"Part0" + str(i) + "/*.bmp")

        for bmpFile in bmpFiles:
            X_test.append(fe.ReadImage(bmpFile))
            Y_test.append(fe.ExtractClass(bmpFile))
    print(len(Y_test))
    return X_test, Y_test

def prepare_english_test():
    X_test = []
    Y_test = []

    test_path = "Data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
    label_path = "Data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"

    Y_test = idx2numpy.convert_from_file(label_path)
    X_test = np.reshape((idx2numpy.convert_from_file(test_path))/255,(len(Y_test),-1))
    return X_test,Y_test