import DataSet_Preparation as pre
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def SVM_test(svclassifier):
    print("Preparing test array")
    X_test,Y_test=pre.prepare_test()
    print("SVM Classifier Prediction")
    Y_pred = svclassifier.predict(X_test)
    print("Accuracy of Arabic Data set: ", np.count_nonzero(np.array(Y_pred)==np.array(Y_test))/len(Y_test)*100, "%")

def SVM_english_test(svclassifier):
    print("Preparing test array")
    X_test,Y_test=pre.prepare_english_test()
    print("SVM Classifier Prediction")
    Y_pred = svclassifier.predict(X_test)
    print("Accuracy of English Data set: ", np.count_nonzero(np.array(Y_pred)==np.array(Y_test))/len(Y_test)*100, "%")
    print("Precision Score: ", precision_score(Y_test, Y_pred,average='micro')*100,"%")
    print("Recall Score: ", recall_score(Y_test, Y_pred,average='micro')*100,"%")
    print("F-1 Score: ", f1_score(Y_test, Y_pred,average='micro')*100,"%")
