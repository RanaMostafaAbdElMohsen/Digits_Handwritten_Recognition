from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import DataSet_Preparation as pre
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def test_arabic_CNN(model):
    X_test, Y_test = pre.prepare_test()
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
    Y_test = np_utils.to_categorical(Y_test)
    Y_pred = model.predict(X_test)
    Y_Predict = []
    Y_Label = []
    for i in range(Y_test.shape[0]):
        Y_Predict.append(np.argmax(Y_pred[i]))
        Y_Label.append(np.argmax(Y_test[i]))
    print("Accuracy of CNN Arabic Model: ",accuracy_score(Y_Predict,Y_Label) * 100,"%")
    print("Precision Score: ", precision_score(Y_Predict,Y_Label, average='micro') * 100, "%")
    print("Recall Score: ", recall_score(Y_Predict,Y_Label, average='micro') * 100, "%")
    print("F-1 Score: ", f1_score(Y_Predict,Y_Label, average='micro') * 100, "%")

def test_english_CNN(model):
    X_test, Y_test = pre.prepare_english_test()
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
    Y_test = np_utils.to_categorical(Y_test)
    Y_pred = model.predict(X_test)
    Y_Predict=[]
    Y_Label=[]
    for i in range(Y_test.shape[0]):
        Y_Predict.append(np.argmax(Y_pred[i]))
        Y_Label.append(np.argmax(Y_test[i]))
    print("Accuracy of CNN English Model: ",accuracy_score(Y_Predict,Y_Label) * 100,"%")
    print("Precision Score: ", precision_score(Y_Predict,Y_Label, average='micro') * 100, "%")
    print("Recall Score: ", recall_score(Y_Predict,Y_Label, average='micro') * 100, "%")
    print("F-1 Score: ", f1_score(Y_Predict,Y_Label, average='micro') * 100, "%")