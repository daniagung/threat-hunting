import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.externals import joblib


data = pd.read_csv('command.csv').values
X = data[:,0:49]
y = data[:,50]



pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
plt.scatter(principalComponents[:,0],principalComponents[:,1],c=y)
plt.show()


kf = KFold(n_splits=10)
print(kf)

daftarakurasi = []
daftarpresisi = []
daftarrecall = []
daftarfmeasure = []


clf = MLPClassifier(hidden_layer_sizes=100,  learning_rate_init=0.1, activation='relu', max_iter=100, solver='adam', alpha=0.0001, batch_size='auto')

for train_index, test_index in kf.split(X): 
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pred = clf.fit(X_train,y_train).predict(X_test)
    
    loss = clf.loss_curve_
    for i in range(len(loss)):
        loss[i] = 1-loss[i]
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.plot(loss)
    plt.show()
    
    akurasi = accuracy_score(y_test,pred) * 100
    presisi = precision_score(y_test,pred) * 100
    recall = recall_score(y_test,pred) * 100
    f_measure = f1_score(y_test,pred) * 100
   
    
    daftarakurasi.append(akurasi)
    daftarpresisi.append(presisi)
    daftarrecall.append(recall)
    daftarfmeasure.append(f_measure)
    
  

    print ("Daftar Akurasi =", daftarakurasi)
    print ("Daftar Presisi =", daftarpresisi)
    print ("Daftar Recall =", daftarrecall)
    print ("Daftar F-Measure =", daftarfmeasure)
    loss = clf.loss_curve_
    for i in range(len(loss)):
        print("Iterasi = %d, Akurasi = %f, Loss = %f" % (i+1, 1-loss[i], loss[i]))
        print("===================================================================\n")
    

#Mencari Rata-Rata Akurasi
def rata2(arraynya):
    return sum(arraynya)/len(arraynya)

print("=========================================\n")
print("Akurasi akhir =", rata2(daftarakurasi))
print("Presisi akhir =", rata2(daftarpresisi))
print("Recall akhir =", rata2(daftarrecall))
print("F Measure akhir =", rata2(daftarfmeasure))
print("=========================================\n")



plt.plot(clf.loss_curve_)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.1))
plt.show()

# save the model to disk
filename = 'lateralmovement.sav'
joblib.dump(clf, filename)