# Klasifikasi karakter tulisan tangan pada dataset MNIST menggunakan HOG Feature Extraction dan Support Vector Machine (SVM)

# Import library
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import datasets # Library untuk membangun dan melatih model pembelajaran menggunakan TensorFlow
from skimage.feature import hog # Library untuk ekstraksi fitur HOG (Histogram of Oriented Gradients)
from sklearn.svm import SVC # Library untuk klasifikasi dengan Support Vector Machine (SVM)
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix # Library untuk mengevaluasi performa model pembelajaran 

# Load data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() # Meload data dari dataset MNIST

# Ekstraksi Fitur HOG untuk data latih
hog_features_train = [] # Inisialisasi variabel untuk fitur HOG dari data latih
for image in x_train: # Loop untuk setiap citra dalam data latih
    fd = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) # Ekstraksi fitur HOG dari sebuah citra
    hog_features_train.append(fd) # Menambahkan fitur HOG dari citra ke dalam list hog_features_train.

hog_features_train = np.array(hog_features_train) # Mengubah hog_features_train menjadi array numpy

# Ekstraksi fitur HOG untuk data test
hog_features_test = [] # Inisialisasi variabel untuk fitur HOG dari data test
for image in x_test: # Loop untuk setiap citra dalam data test
    fd = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) # Ekstraksi fitur HOG dari sebuah citra
    hog_features_test.append(fd)  # Menambahkan fitur HOG dari citra ke dalam list hog_features_test.
    
hog_features_test = np.array(hog_features_test) # Mengubah hog_features_test menjadi array numpy

# Membuat model SVM dan melatihnya
svm_model = SVC(gamma='scale') # Membuat instance model SVM dari sklearn dengan parameter gamma='scale'
svm_model.fit(hog_features_train, y_train) # Proses pelatihan model SVM dengan data fitur dan label dari dataset latih

# Prediksi dan evaluasi
svm_predictions = svm_model.predict(hog_features_test) # Memprediksi dataset test menggunakan model SVM yang sudah dilatih

# Menampilkan hasil evaluasi
print("Confusion Matrix:", confusion_matrix(y_test, svm_predictions)) # Menampilkan hasil prediksi confusion matrix
print("Accuracy:", accuracy_score(y_test, svm_predictions)) # Menampilkan hasil prediksi accuracy
print("Precision:", precision_score(y_test, svm_predictions, average='weighted')) # Menampilkan hasil prediksi precision