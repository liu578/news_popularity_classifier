##################################################
## EE559 project, Spring 2018
## Created by Yang Liu,liu578@usc.edu
## Tested in Python 3.5.4, OSX 10.13.4
## Used scikit-learn and Python
## usage:uncomment any classifier in main function
###################################################

import numpy as np
np.set_printoptions(suppress=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

fp = open('OnlineNewsPopularityReduced.csv')
features = fp.readline().strip('\n').split(",")
fp.close()

data_raw = np.genfromtxt('OnlineNewsPopularityReduced.csv', delimiter=',')
nPoint = data_raw.shape[0] - 1 #4954
nFeature = data_raw.shape[1] - 2 #59
# url and timedelta is not considered as useful feature
data_un_nom = data_raw[1:, 2:data_raw.shape[1]]
# totally 59 featureas here and shares is included

threshold = 1400
nClass1 = 0
nClass2 = 0
labels = [0] * nPoint

#1 ***********Labeling Dataset ***********
for i in range(nPoint):
    datapoint = data_un_nom[i]
    shares_value = datapoint[nFeature - 1]
    if shares_value <= threshold:
        labels[i] = 2
        nClass2 += 1
    else:  # shares_value > threshold:
        labels[i] = 1
        nClass1 += 1

#2 ***********Separating Training and Test Set ***********
train_data_un_nom, test_data_un_nom, train_labels, test_labels = train_test_split(data_un_nom, np.asarray(labels),test_size=0.19257, random_state=1)

#3 ***********Traverse the training set and calculate Fisher Score***********
labels_of_train = [0] * 4000
nClass1_of_train = 0
nClass2_of_train = 0
sum1 = [0.00] * nFeature
sum2 = [0.00] * nFeature
u1 = [0.00] * nFeature
u2 = [0.00] * nFeature
s1 = [0.00] * nFeature
s2 = [0.00] * nFeature
F_score = [0.00] * nFeature
for i in range(4000):
    datapoint = train_data_un_nom[i]
    shares_value = datapoint[nFeature - 1]
    if shares_value <= threshold:
        labels_of_train[i] = 2
        nClass2_of_train += 1
        sum2 = sum2 + datapoint
    else:  # shares_value > threshold:
        labels_of_train[i] = 1
        nClass1_of_train += 1
        sum1 = sum1 + datapoint

u1 = sum1 / nClass1_of_train
u2 = sum2 / nClass2_of_train
for j in range(nFeature):
    for i in range(4000):
        if labels_of_train[i] == 1:
            s1[j] += (train_data_un_nom[i][j] - u1[j]) * (train_data_un_nom[i][j] - u1[j])
        else:
            s2[j] += (train_data_un_nom[i][j] - u2[j]) * (train_data_un_nom[i][j] - u2[j])
    if j == 17:
        continue
    F_score[j] = ((u1[j] - u2[j]) * (u1[j] - u2[j])) / (s1[j] + s2[j])

#4 ***********Normalization ***********
scaler = StandardScaler()
scaler.fit(train_data_un_nom)
train_data = scaler.transform(train_data_un_nom)
# standardize testing data
test_data = scaler.transform(test_data_un_nom)
# print(train_data.shape,test_data.shape)

#1 ***********PCA ***********
pca = PCA(n_components=20)
train_data_pca = pca.fit_transform(train_data[:, 0:train_data.shape[1] - 1])  # remove shares feature ,use 58(all) features
test_data_pca = pca.transform(test_data[:, 0:test_data.shape[1] - 1])

#1 ***********fisher score ***********
tup = zip(F_score, range(len(F_score)))
sorted_list = sorted(tup, key=lambda v: v[0], reverse=True)
# print(sorted_list)
F_indexes = []
for key, value in sorted_list:
    print(value + 2, features[value + 2])
    F_indexes.append(value + 2)
# use 20 features here
indexes = (F_indexes[1:21])
print("index of features chosen by Fisher Score:",indexes)
print('-------------------------')

def knn_5():
    data_knn = train_data[:, 0:train_data.shape[1] - 1]  # remove shares feature ,use 58(all) features
    save_avg_acc = []
    for epch in range(0, 3):
        # print('in epch {:d}'.format(epch))
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        cnt = 1;
        for train_index, dev_index in skf.split(data_knn, train_labels):
            X_cv_train, X_cv_dev = data_knn[train_index], data_knn[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm="auto")
            knn_clf.fit(X_cv_train, y_cv_train)
            y_pred = knn_clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}%'.format(100*acc))
            all_acc.append(acc)
            cnt = cnt + 1
        mean_ = np.mean(all_acc)
        # print('Average 5 fold cross valication accuracy = {:.2f}%'.format(100*mean_))
        save_avg_acc.append(mean_)
    print('5NN training accuracy: {:.2f}%'.format(100*np.mean(save_avg_acc)))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="auto")
    knn_clf.fit(data_knn, train_labels)
    # print(data_knn.shape,test_data[:, 0:test_data.shape[1] - 1].shape)
    # now test
    y_test_pred = knn_clf.predict(test_data[:, 0:test_data.shape[1] - 1])
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('5NN testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def knn_5_pca():
    data_knn = train_data_pca
    save_avg_acc = []
    for epch in range(0, 3):
        # print('in epch {:d}'.format(epch))
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        cnt = 1;
        for train_index, dev_index in skf.split(data_knn, train_labels):
            X_cv_train, X_cv_dev = data_knn[train_index], data_knn[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm="auto")
            knn_clf.fit(X_cv_train, y_cv_train)
            y_pred = knn_clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}%'.format(100*acc))
            all_acc.append(acc)
            cnt = cnt + 1
        mean_ = np.mean(all_acc)
        # print('Average 5 fold cross valication accuracy = {:.2f}%'.format(100*mean_))
        save_avg_acc.append(mean_)
    print('5nn training accuracy: {:.2f}%'.format(100*np.mean(save_avg_acc)))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="auto")
    knn_clf.fit(data_knn, train_labels)
    # print(data_knn.shape,test_data_pca.shape)
    # now test
    y_test_pred = knn_clf.predict(test_data_pca)
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('5nn testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def knn_5_fisher():
    data_knn = train_data[:][:, indexes]
    save_avg_acc = []
    for epch in range(0, 3):
        # print('in epch {:d}'.format(epch))
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        cnt = 1;
        for train_index, dev_index in skf.split(data_knn, train_labels):
            X_cv_train, X_cv_dev = data_knn[train_index], data_knn[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5,algorithm="auto")
            knn_clf.fit(X_cv_train, y_cv_train)
            y_pred = knn_clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}%'.format(100*acc))
            all_acc.append(acc)
            cnt = cnt + 1
        mean_ = np.mean(all_acc)
        # print('Average 5 fold cross valication accuracy = {:.2f}%'.format(100*mean_))
        save_avg_acc.append(mean_)
    print('5nn training accuracy: {:.2f}%'.format(100*np.mean(save_avg_acc)))
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="auto")
    knn_clf.fit(data_knn, train_labels)
    # print(data_knn.shape,test_data[:][:, indexes].shape)
    # now test
    y_test_pred = knn_clf.predict(test_data[:][:, indexes])
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('5nn testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def knn_fisher():
    data_knn = train_data[:][:, indexes]
    save_avg_acc = []
    acc_k=[]
    for k in range(3, 12):
        for epch in range(0, 5):

            skf = StratifiedKFold(n_splits=5, shuffle=True)
            all_acc = []
            cnt = 1
            for train_index, dev_index in skf.split(data_knn, train_labels):
                X_cv_train, X_cv_dev = data_knn[train_index], data_knn[dev_index]
                y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
                knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k,algorithm="auto")
                knn_clf.fit(X_cv_train, y_cv_train)
                y_pred = knn_clf.predict(X_cv_dev)
                acc = accuracy_score(y_cv_dev, y_pred)
                # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}%'.format(100*acc))
                all_acc.append(acc)
                cnt += 1
            mean_ = np.mean(all_acc)
            # print('Average 5 fold cross valication accuracy = {:.2f}%'.format(100*mean_))
            save_avg_acc.append(mean_)
        mean_k = np.mean(save_avg_acc)
        acc_k.append(mean_k)
    # choose best value of k
    indx = np.argmax(acc_k)
    print('Best KNN training accuracy :  {:.2f}%'.format(100* acc_k[indx]), 'best k =', indx + 3)
    # use this best k to train the whole training set
    # because we want to train the whole validable training set
    best_clf = neighbors.KNeighborsClassifier(n_neighbors=indx + 3,algorithm="auto")
    best_clf.fit(data_knn, train_labels)
    # print(data_knn.shape,test_data[:][:, indexes].shape)
    # now test
    y_test_pred = best_clf.predict(test_data[:][:, indexes])
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('KNN testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def knn_pca():
    data_knn = train_data_pca
    save_avg_acc = []
    acc_k = []
    for k in range(3, 12):
        for epch in range(0, 5):

            skf = StratifiedKFold(n_splits=5, shuffle=True)
            all_acc = []
            cnt = 1
            for train_index, dev_index in skf.split(data_knn, train_labels):
                X_cv_train, X_cv_dev = data_knn[train_index], data_knn[dev_index]
                y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
                knn_clf = neighbors.KNeighborsClassifier(n_neighbors=k,algorithm="auto")
                knn_clf.fit(X_cv_train, y_cv_train)
                y_pred = knn_clf.predict(X_cv_dev)
                acc = accuracy_score(y_cv_dev, y_pred)
                # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}%'.format(100*acc))
                all_acc.append(acc)
                cnt += 1
            mean_ = np.mean(all_acc)
            # print('Average 5 fold cross valication accuracy = {:.2f}%'.format(100*mean_))
            save_avg_acc.append(mean_)
        mean_k = np.mean(save_avg_acc)
        acc_k.append(mean_k)
    # choose best value of k
    indx = np.argmax(acc_k)
    print('Best KNN training accuracy :  {:.2f}%'.format(100* acc_k[indx]), 'best k =', indx + 3)
    # use this best k to train the whole training set
    # because we want to train the whole validable training set
    best_clf = neighbors.KNeighborsClassifier(n_neighbors=indx + 3,algorithm="auto")
    best_clf.fit(data_knn, train_labels)
    # print(data_knn.shape,test_data_pca.shape)
    # now test
    y_test_pred = best_clf.predict(test_data_pca)
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('KNN testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def lin_kernal():
    data_svm = train_data[:][:, indexes]

    # dis 10
    Cs = np.logspace(-2, 2, 10)
    # print(Cs)
    save_abg_acc = []
    for i in range(0, 10):
        C = Cs[i]
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        cnt = 1
        for train_index, dev_index in skf.split(data_svm, train_labels):
            X_cv_train, X_cv_dev = data_svm[train_index], data_svm[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            clf = SVC(kernel='linear', C=C)
            clf.fit(X_cv_train, y_cv_train)
            y_pred = clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}'.format(acc))
            all_acc.append(acc)
            cnt += 1
        mean_ = np.mean(all_acc)
        # print('Average 5 fold cross valication accuracy = {:.2f}'.format(mean_))
        save_abg_acc.append(mean_)

    # plt.plot(Cs, save_abg_acc)
    # plt.show()

    # choose best value of C
    indx = np.argmax(save_abg_acc)
    print('Best SVM training accuracy: {:.2f}% '.format(100* save_abg_acc[indx]), 'best C =', Cs[indx])
    # use this best C to train the whole training set
    # because we want to train the whole validable training set
    best_clf = SVC(kernel='linear', C=Cs[indx])
    best_clf.fit(data_svm, train_labels)
    # print(data_svm.shape,test_data[:][:, indexes].shape)

    # now test
    y_test_pred = best_clf.predict(test_data[:][:, indexes])
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('SVM testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def rbf_kernel():
    data_svm = train_data[:][:, indexes]
    Cs = np.logspace(-2, 2, 5)
    rs = np.logspace(-2, 2, 5)
    save_avg_acc = np.zeros((5, 5))
    save_avg_std = np.zeros((5, 5))
    for i in range(0, 5):
        for j in range(0, 5):
            C = Cs[i]
            r = rs[j]
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            all_acc = []
            # cnt = 1
            for train_index, dev_index in skf.split(data_svm, train_labels):
                X_cv_train, X_cv_dev = data_svm[train_index], data_svm[dev_index]
                y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
                clf = SVC(kernel='rbf', gamma=r, C=C)
                clf.fit(X_cv_train, y_cv_train)
                y_pred = clf.predict(X_cv_dev)
                acc = accuracy_score(y_cv_dev, y_pred)
                # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}'.format(acc))
                all_acc.append(acc)
                # cnt+= 1
            mean_ = np.mean(all_acc)
            std_ = np.std(all_acc)
            # print(i,j,'Average 5 fold cross valication accuracy = {:.2f}'.format(mean_))
            # print('Standard deviation of 5 fold cross valication accuracy = {:.2f}'.format(std_))
            save_avg_acc[i][j] = mean_
            save_avg_std[i][j] = std_

    # #Visualization

    # X, Y = np.meshgrid(rs, Cs)
    # Z = save_avg_acc
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
    # ax.set_xlabel('gamma ')
    # ax.set_ylabel('C ')
    # ax.set_zlabel('average accuracies ')
    # plt.show()


    # choose best value of C and gamma
    m, n = np.where(save_avg_acc == save_avg_acc.max())
    if (m.size > 1):
        k = np.argmin(save_avg_std[m, n])
        m = m[k]
        n = n[k]

    print('best gamma = ', rs[n][0])
    print('best C = ', Cs[m][0], )
    print('SVM training accuracy  : {:.2f}%'.format(100* save_avg_acc[m, n][0]))
    print('standard deviation  = ', save_avg_std[m, n][0])

    # return (Cs[m], rs[n], save_avg_acc[m, n], save_avg_std[m, n])


    best_clf = SVC(kernel='rbf', gamma=rs[n][0], C=Cs[m][0])
    best_clf.fit(data_svm, train_labels)
    # print(data_svm.shape,test_data[:][:, indexes].shape)

    # # now test
    y_test_pred = best_clf.predict(test_data[:][:, indexes])
    # print(test_labels.shape, y_test_pred.shape)
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('SVM testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))

def svm_fisher():
    lin_kernal()
    rbf_kernel()


def lin_kernal_pca():
    data_svm = train_data_pca

    # dis 10
    Cs = np.logspace(-2, 2, 10)
    # print(Cs)
    save_abg_acc = []
    for i in range(0, 10):
        C = Cs[i]
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        cnt = 1
        for train_index, dev_index in skf.split(data_svm, train_labels):
            X_cv_train, X_cv_dev = data_svm[train_index], data_svm[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            clf = SVC(kernel='linear', C=C)
            clf.fit(X_cv_train, y_cv_train)
            y_pred = clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}'.format(acc))
            all_acc.append(acc)
            cnt += 1
        mean_ = np.mean(all_acc)
        # print('Average 5 fold cross valication accuracy = {:.2f}'.format(mean_))
        save_abg_acc.append(mean_)

    # plt.plot(Cs, save_abg_acc)
    # plt.show()

    # choose best value of C
    indx = np.argmax(save_abg_acc)
    print('Best SVM training accuracy: {:.2f}% '.format(100* save_abg_acc[indx]), 'best C =', Cs[indx])

    # use this best C to train the whole training set
    # because we want to train the whole validable training set
    best_clf = SVC(kernel='linear', C=Cs[indx])
    best_clf.fit(data_svm, train_labels)
    # print(data_svm.shape,test_data[:][:, indexes].shape)

    # now test
    y_test_pred = best_clf.predict(test_data_pca)
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('SVM testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def rbf_kernel_pca():
    data_svm = train_data_pca
    Cs = np.logspace(-2, 2, 5)
    rs = np.logspace(-2, 2, 5)
    save_avg_acc = np.zeros((5, 5))
    save_avg_std = np.zeros((5, 5))
    for i in range(0, 5):
        for j in range(0, 5):
            C = Cs[i]
            r = rs[j]
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            all_acc = []
            # cnt = 1
            for train_index, dev_index in skf.split(data_svm, train_labels):
                X_cv_train, X_cv_dev = data_svm[train_index], data_svm[dev_index]
                y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
                clf = SVC(kernel='rbf', gamma=r, C=C)
                clf.fit(X_cv_train, y_cv_train)
                y_pred = clf.predict(X_cv_dev)
                acc = accuracy_score(y_cv_dev, y_pred)
                # print('Accuracy in fold {:d}'.format(cnt), ' = {:.2f}'.format(acc))
                all_acc.append(acc)
                # cnt+= 1
            mean_ = np.mean(all_acc)
            std_ = np.std(all_acc)
            # print(i,j,'Average 5 fold cross valication accuracy = {:.2f}'.format(mean_))
            # print('Standard deviation of 5 fold cross valication accuracy = {:.2f}'.format(std_))
            save_avg_acc[i][j] = mean_
            save_avg_std[i][j] = std_

    # #Visualization

    # X, Y = np.meshgrid(rs, Cs)
    # Z = save_avg_acc
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
    # ax.set_xlabel('gamma ')
    # ax.set_ylabel('C ')
    # ax.set_zlabel('average accuracies ')
    # plt.show()


    # choose best value of C and gamma
    m, n = np.where(save_avg_acc == save_avg_acc.max())
    if (m.size > 1):
        k = np.argmin(save_avg_std[m, n])
        m = m[k]
        n = n[k]

    print('best gamma = ', rs[n][0])
    print('best C = ', Cs[m][0], )
    print('SVM training accuracy  : {:.2f}%'.format(100* save_avg_acc[m, n][0]))
    print('standard deviation  = ', save_avg_std[m, n][0])

    # return (Cs[m], rs[n], save_avg_acc[m, n], save_avg_std[m, n])


    best_clf = SVC(kernel='rbf', gamma=rs[n][0], C=Cs[m][0])
    best_clf.fit(data_svm, train_labels)
    # print(data_svm.shape,test_data[:][:, indexes].shape)

    # # now test
    y_test_pred = best_clf.predict(test_data_pca)
    # print(test_labels.shape, y_test_pred.shape)
    test_acc = accuracy_score(test_labels, y_test_pred)
    print('SVM testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def svm_pca():
    lin_kernal_pca()
    rbf_kernel_pca()


def ann_fisher():
    data_ann = train_data[:][:, indexes]
    # data_ann = train_data[:, 0:train_data.shape[1]-1]# remove shares feature ,use 58(all) features
    save_avg_acc = []
    # layers = [(100), (1000, 100), (10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10, 10)]
    layers = [(100),(100,100),(10, 10, 10), (10, 10, 10, 10),(10, 10, 10, 10, 10)]
    for index_of_layer in range(len(layers)):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        for train_index, dev_index in skf.split(data_ann, train_labels):
            X_cv_train, X_cv_dev = data_ann[train_index], data_ann[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            # mlp_clf = MLPClassifier(hidden_layer_sizes=layers[index_of_layer], activation="tanh",solver="lbfgs")  # most high
            mlp_clf = MLPClassifier(hidden_layer_sizes=layers[index_of_layer], activation="identity", solver="lbfgs")
            mlp_clf.fit(X_cv_train, y_cv_train)
            y_pred = mlp_clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            all_acc.append(acc)
        mean_ = np.mean(all_acc)
        print('hidden_layer ',layers[index_of_layer],'Average training accuracy = {:.2f}%'.format(100*mean_))
        save_avg_acc.append(mean_)

        # for epch in range(10):

    # choose best value of k
    indx = np.argmax(save_avg_acc)
    print('Best MLP training accuracy = {:.2f}%'.format(100*save_avg_acc[indx]), 'best hidden_layer =', layers[indx])
    # use this best hidden_layer to train the whole training set
    # because we want to train the whole validable training set
    best_clf = MLPClassifier(hidden_layer_sizes=layers[indx],activation="identity", solver="lbfgs")
    best_clf.fit(data_ann, train_labels)

    # now test
    y_test_pred = best_clf.predict(test_data[:][:, indexes])
    # y_test_pred = best_clf.predict(test_data[:, 0:test_data.shape[1]-1])

    test_acc = accuracy_score(test_labels, y_test_pred)
    print('MLP testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def ann_pca():
    data_ann = train_data_pca
    # data_ann = train_data[:, 0:train_data.shape[1]-1]# remove shares feature ,use 58(all) features
    save_avg_acc = []
    # layers = [(100), (1000, 100), (10, 10, 10), (10, 10, 10, 10), (10, 10, 10, 10, 10)]
    layers = [(100),(100,100),(10, 10, 10), (10, 10, 10, 10),(10, 10, 10, 10, 10)]
    for index_of_layer in range(len(layers)):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        all_acc = []
        for train_index, dev_index in skf.split(data_ann, train_labels):
            X_cv_train, X_cv_dev = data_ann[train_index], data_ann[dev_index]
            y_cv_train, y_cv_dev = train_labels[train_index], train_labels[dev_index]
            # mlp_clf = MLPClassifier(hidden_layer_sizes=layers[index_of_layer], activation="tanh",solver="lbfgs")  # most high
            mlp_clf = MLPClassifier(hidden_layer_sizes=layers[index_of_layer], activation="identity", solver="lbfgs")
            mlp_clf.fit(X_cv_train, y_cv_train)
            y_pred = mlp_clf.predict(X_cv_dev)
            acc = accuracy_score(y_cv_dev, y_pred)
            all_acc.append(acc)
        mean_ = np.mean(all_acc)
        print('hidden_layer ',layers[index_of_layer],'Average training accuracy = {:.2f}%'.format(100*mean_))
        save_avg_acc.append(mean_)


        # for epch in range(10):

    # choose best value of k
    indx = np.argmax(save_avg_acc)
    print('Best MLP accuracy = {:.2f}%'.format(100*save_avg_acc[indx]), 'best hidden_layer =', layers[indx])
    # use this best hidden_layer to train the whole training set
    # because we want to train the whole validable training set
    best_clf = MLPClassifier(hidden_layer_sizes=layers[indx],activation="identity", solver="lbfgs")
    best_clf.fit(data_ann, train_labels)

    # now test
    y_test_pred = best_clf.predict(test_data_pca)
    # y_test_pred = best_clf.predict(test_data[:, 0:test_data.shape[1]-1])

    test_acc = accuracy_score(test_labels, y_test_pred)
    print('MLP testing accuracy : {:.2f}%'.format(100*test_acc))
    print(classification_report(test_labels, y_test_pred,target_names = ['class 1 popular','class 2 nonpopular']))
    print('confusion matrix :\n',confusion_matrix(test_labels, y_test_pred))


def main():
    print('***********5nn***********')
    # knn_5()
    # knn_5_pca()
    # knn_5_fisher()

    print('***********KNN***********')
    # knn_pca()
    # knn_fisher()
    print('***********SVM***********')
    # svm_pca()
    # lin_kernal_pca()
    # rbf_kernel_pca()

    # svm_fisher()
    # lin_kernal()
    rbf_kernel()
    print('***********MLP***********')
    # ann_pca()
    # ann_fisher()
if __name__ == '__main__':
    main()
