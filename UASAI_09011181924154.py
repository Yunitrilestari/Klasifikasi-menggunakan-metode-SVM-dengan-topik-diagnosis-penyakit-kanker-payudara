#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi Supervised Learning menggunakan metode Support Vector Machine (SVM) dengan topik Mediacal Diagnosis penyakit kanker Payudara.¶

# Pada penelitian kali ini saya menggunakan metode SVM karena dalam hal pengklasifikasian SVM cukup baik dibandingankan dengan metode klasifikasi lainnya.
# 
# Dan pada penelitian ini pula saya menggunakan dataset yang saya dapatkan dari UCI yaitu 
# Breast Cancer Wisconsin (Diagnostic) Data Set.
# link : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# 
# Dataset ini dihitung dari gambar digital aspirasi jarum halus (FNA) dari massa payudara. Mereka menggambarkan karakteristik inti sel yang ada dalam gambar.
# 
# Dataset terdiri dari 30 fitur (mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error, smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness, worst concavity, worst concave points, worst symmetry, and worst fractal dimension) and a target (type of cancer).
# 
# Data ini memiliki dua jenis kelas kanker: ganas (berbahaya/M) dan jinak (tidak berbahaya/B). 
# Di sini, saya akan membuat model untuk mengklasifikasikan jenis kanker.

# # 1.Penguploadan data

# Pada tahap pertama ini, saya akan mengupload data yang akan digunakan dalam penelitian ini.

# In[6]:


#Impor library yang digunakan yaitu sklearm dan datasets
from sklearn import datasets

#Memuat data , yang mana disimpan pada variabel cancer
cancer = datasets.load_breast_cancer()


# # 2.  Explore data 

# Pada tahap kedua saya melakukan explore data untuk mengetahui lebih banyak informasi tentang data,seperti variabel yang tersedia , jumlah baris dan kolom pada data, 5 data kanker teratas dan target.

# In[7]:


# Mencetak 30 label pada kolom
print("Features: ", cancer.feature_names)

# Mencetak label jenis kanker('malignant' 'benign')
print("Labels: ", cancer.target_names)


# In[8]:


# Menampilkan jumlah baris dan kolom pada dataset
cancer.data.shape
#Dataset terdiri dari 569 baris dan 30 kolom.


# In[9]:


# menampilkan fitur 5 data kanker teratas
print(cancer.data[0:5])


# In[10]:


#Menampilkan label kanker pada dataset (0:malignant, 1:benign)
print(cancer.target)


# # 3. Memisahkan data

# Tahap ketiga adalah memisahkan data menjadi 2 yaitu data training dan data testing. Jika dilihat pada program maka
# data training yang digunakan adalah sebanyak 70% dan data testing sebanyak 30%.

# In[11]:


# Impor fungsi train_test_split
from sklearn.model_selection import train_test_split

# Memisahkan dataset menjadi data training dan set testing
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


# # 4. Membuat model

# Kemudian, kita akan membuat model yang akan digunakan untuk melakukan klasifikasi serta melakukan prediksi terhadap data training.

# In[12]:


#Import svm model
from sklearn import svm

#Membuat Pengklasifikasi svm
clf = svm.SVC(kernel='linear') # Linear Kernel

#Melatih model menggunakan data training
clf.fit(X_train, y_train)

#Prediksi respons untuk kumpulan data training
y_pred = clf.predict(X_test)


# # 5. Mengevaluasi model

# Pada tahap kelima kita akan mengevaluasi model yang kita gunakan dengan mencari akurasinya, Akurasi dapat dihitung dengan 
# membandingkan nilai dari data training aktual dan nilai prediksi.

# In[13]:


#Impor modul metrik scikit-learn untuk perhitungan akurasi
from sklearn import metrics

# Model Accuracy: seberapa sering klasifikasi benar?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Nah, kita mendapatkan nilai akurasi sebesar 96,49%, dan dianggap sebagai akurasi yang sangat baik.

# In[14]:


#Melihat confusion matrix untuk memudahkan dalam mengetahui apakah terdapat kesalahan dalam pengklasifikasian
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# Berdasarkan output, diketahui bahwa terdapat 2 kesalahan dalam pengklasifikasian menggunakan algoritma SVM, 
# yaitu 2 pasien diklasifikasikan dalam pasien pengidap kanker jinak, tetapi dalam keadaan sebenarnya, pasien mengalami kanker ganas dan 4 pasien diklasifikasikan dalam pasien pengidap kanker ganas, tetapi dalam keadaan sebenarnya, pasien mengalami kanker jinak. 

# Untuk evaluasi lebih lanjut, kita juga dapat memeriksa presisi dan recall model.

# In[15]:


# Model Precision: berapa persentase tupel positif yang diberi label seperti itu?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: berapa persentase tupel positif yang diberi label seperti itu?
print("Recall:",metrics.recall_score(y_test, y_pred))


# Nah, kita mendapatkan presisi 98% dan recall 96%, yang dianggap sebagai nilai yang sangat baik.
