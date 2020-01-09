import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris

# データセットの生成
#X, y = mglearn.datasets.make_forge()
# データセットをプロット
#mglearn.discrete_scatter(X[:,0], X[:,1], y)
#plt.legend(["Class 0", "Class 1"], loc=4)
#plt.xlabel("First feature")
#plt.ylabel("Second feature")
#print("X.shape : {}".format(X.shape))
#plt.show()

#X, y = mglearn.datasets.make_wave(n_samples=50)
#plt.plot(X, y, "o")
#plt.ylim(-3, 3)
#plt.xlabel("Feature")
#plt.ylabel("Target")
#plt.show()

# 腫瘍が悪性かどうかを判断する
#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()
#print("cancer.keys() : \n{}".format(cancer.keys()))
#print("Shape of cancer data: {}".format(cancer.data.shape))
#print("Sample counts per class:\n{}".format(
#    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
#print("Feature names:\n{}".format(cancer.feature_names))


# ボストンの住宅価格
#from sklearn.datasets import load_boston
#boston = load_boston()
#print("Data shape: {}".format(boston.data.shape))


# ボストンの住宅価格（特徴量間の積も含める）
#X, y = mglearn.datasets.load_extended_boston()
#print("X.shape: {}".format(X.shape))
#mglearn.plots.plot_knn_classification(n_neighbors=3)
#plt.show()


# p55
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

#print("Test set predictions: {}".format(clf.predict(X_test)))
#print("Test set accuracy : {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # fitメソッドは自分自身を返すので、1行で
    # インスタンスを生成してfitすることができる
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)


plt.show()