import train_SVM as sv_tr
import test_SVM as sv_te
import train_CNN as cnn_tr
import test_CNN as cnn_te

def run_svm_arabic_set():
    svm_classifier = sv_tr.SVM_arabic_train()
    sv_te.SVM_test(classifier)

def run_svm_english_set():
    svm_classifier = sv_tr.SVM_english_train()
    sv_te.SVM_english_test(svm_classifier)

def run_cnn_arabic_set():
    cnn_classifier= cnn_tr.train_arabic_CNN()
    cnn_te.test_arabic_CNN(cnn_classifier)

def run_cnn_english_set():
    cnn_classifier= cnn_tr.train_english_CNN()
    cnn_te.test_english_CNN(cnn_classifier)

def run():
    print("###############################################################################################")
    print("Arabic Data Set:")
    print("SVM Classifier: ")
    # run_svm_arabic_set()
    print("CNN Classifier: ")
    # run_cnn_arabic_set()
    print("###############################################################################################")
    print("English Data Set:")
    print("SVM Classifier: ")
    # run_svm_english_set()
    print("CNN Classifier: ")
    run_cnn_english_set()

run()