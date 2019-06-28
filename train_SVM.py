import DataSet_Preparation as pre
from sklearn.svm import SVC


def SVM_arabic_train():
    X_train=[]
    Y_train=[]
    print("Preparing train array")
    X_train, Y_train = pre.prepare_train()
    print("SVM Classifier Training")
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    return svclassifier

def SVM_english_train():

    print("Preparing train array")
    X_train, Y_train = pre.prepare_english_train()
    print("SVM Classifier Training")
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, Y_train)
    return svclassifier
