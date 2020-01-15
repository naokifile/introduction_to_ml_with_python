import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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
#from sklearn.model_selection import train_test_split
#X, y = mglearn.datasets.make_forge()
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=3)
#clf.fit(X_train, y_train)

#print("Test set predictions: {}".format(clf.predict(X_test)))
#print("Test set accuracy : {:.2f}".format(clf.score(X_test, y_test)))

#fig, axes = plt.subplots(1, 3, figsize=(10, 3))

#for n_neighbors, ax in zip([1, 3, 9], axes):
    # fitメソッドは自分自身を返すので、1行で
    # インスタンスを生成してfitすることができる
#    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{} neighbor(s)".format(n_neighbors))
#    ax.set_xlabel("feature 0")
#    ax.set_ylabel("feature 1")
#axes[0].legend(loc=3)

#plt.show()


# p57
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# n_neighborsを1から10まで試す
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # モデルを構築
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # 訓練セット精度を記録
    training_accuracy.append(clf.score(X_train, y_train))
    # 汎化精度を記録
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
"""


# p58
#mglearn.plots.plot_knn_regression(n_neighbors=1)
#mglearn.plots.plot_knn_regression(n_neighbors=3)
#plt.show()


# p60
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=40)

# waveデータセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 3つの最近傍点を考慮するように設定してモデルのインスタンスを生成
reg = KNeighborsRegressor(n_neighbors=3)
# 訓練データと訓練ターゲットを用いてモデルを学習させる
reg.fit(X_train, y_train)

print("Test set predictions :\n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
"""


# p61
"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=40)

# waveデータセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# -3から3までの間に1,000点のデータポイントを作る
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, "^", c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axes[0].legend(["Model predictions", "Training data/target",
                "test data/target"], loc="best")
plt.show()
"""

# p63
#mglearn.plots.plot_linear_regression_wave()
#plt.show()


# 通常最小二乗法
"""
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#X, y = mglearn.datasets.make_wave(n_samples=60)
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# リッジ回帰
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

lr = LinearRegression().fit(X_train, y_train)

#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intrecept_: {}".format(lr.intercept_))

#print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

#print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

#print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, "s", label="Ridge alpha=1")
plt.plot(ridge10.coef_, "^", label="Ridge alpha=10")
plt.plot(ridge01.coef_, "v", label="Ridge akpla=0.1")

plt.plot(lr.coef_, "o", label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()

"""


#mglearn.plots.plot_ridge_n_samples()
#plt.show()


# lasso
"""
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lasso = Lasso().fit(X_train, y_train)
#print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
#print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# "max_iter"の値を増やしている
# こうしておかないとモデルが、"max_iter"を増やすように警告を発する
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
#print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
#print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))


lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
#print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
#print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
#print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso.coef_, "s", label="Lasso alpha=1")
plt.plot(lasso001.coef_, "^", label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, "v", label="Lasso alpha=0.00001")

plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25,25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
"""


# p74
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

plt.show()

"""

#mglearn.plots.plot_linear_svc_regularization()
#plt.show()

# p77
#from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()
#X_train, X_test, y_train, y_test = train_test_split(
#    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

"""
logreg = LogisticRegression().fit(X_train, y_train)
#print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
#print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
#print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
#print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
#print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
#print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, "o", label="C=1")
plt.plot(logreg100.coef_.T, "^", label="C=100")
plt.plot(logreg001.coef_.T, "v", label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
#plt.hline(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()

for C, marker in zip([0.001, 1, 100], ["o", "^", "v"]):
    lr_l1 = LogisticRegression(C=C, penalty="l1", solver="liblinear").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
        C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
#plt.hline(0, 0, cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()

"""


from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:,0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.show()