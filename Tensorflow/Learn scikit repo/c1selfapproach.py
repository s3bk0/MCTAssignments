import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import tensorflow as tf

iris = datasets.load_iris()
X_iris, Y_iris = iris['data'], iris['target']

# limit dataset to only the first two entries
X, y = X_iris[:, :], Y_iris

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=33)

# standardise results
scaler = preprocessing.StandardScaler().fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)


colors = ['red', 'greenyellow', 'blue']

for i in range(len(colors)):
    xs = xtrain[:, 0][ytrain == i]
    ys = xtrain[:, 1][ytrain == i]
    # plt.scatter(xs, ys, c=colors[i])

plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

classf = SVC(kernel='linear')
clf = Pipeline([('scaler', preprocessing.StandardScaler()),
        ('linear_model', classf)])#SGDClassifier()
clf.fit(xtrain, ytrain)

print(classf.coef_, classf.intercept_)

x_min, x_max = xtrain[:, 0].min() - .5, xtrain[:, 0].max() + .5
y_min, y_max = xtrain[:, 1].min() - .5, xtrain[:, 1].max() + .5
xs = np.arange(x_min, x_max, 0.5)

fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)

for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])

    # plt.scatter(xtrain[:, 0], xtrain[:, 1], c=ytrain,  cmap=plt.cm.prism)

    ys = (-classf.intercept_[i] - xs * classf.coef_[i, 0]) / classf.coef_[i, 1]

    plt.plot(xs, ys)

# print(classf.predict(np.array([[4.7, 3.1]])))
# print(clf.decision_function([[4.7, 3.1]]))

# evaluate results
from sklearn import metrics

ytrain_pred = clf.predict(xtrain)
score = metrics.accuracy_score(ytrain, ytrain_pred)
print(score)

y_pred = clf.predict(xtest)
score = metrics.accuracy_score(ytest, y_pred)
print(score)

print(metrics.classification_report(ytest, y_pred))
print(metrics.confusion_matrix(ytest, y_pred))

# plt.show()

#cross validation
#create composite model
clf = Pipeline([('scaler', preprocessing.StandardScaler()),
        ('linear_model', SGDClassifier())])

cv = KFold(n_splits=5, shuffle=True, random_state=33)
print(cross_val_score(clf, X, y, cv=cv))

#############################

def evaluate_cross_val(clf, X, y, K):
    cv = KFold(n_splits=K, shuffle=True)

    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), np.std(scores, ddof=1)/np.sqrt(K)))

# xtrain, xtest, ytrain, ytest = train_test_split(X_iris, Y_iris, train_size=0.75)

tfmodel = tf.keras.models.Sequential([
                        tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'),
                        tf.keras.layers.Dense(10, activation='relu'),
                        tf.keras.layers.Dense(3)] )
tfmodel.compile('adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

tfmodel.fit(xtrain, ytrain, epochs=250, verbose=0)
probmodel = tf.keras.Sequential([tfmodel, tf.keras.layers.Softmax()])
ypred = probmodel.predict(xtest).argmax(axis=-1)
ypredtrain = probmodel.predict(xtrain).argmax(axis=-1)
print(metrics.accuracy_score(ytrain, ypredtrain))
print(metrics.accuracy_score(ytest, ypred))
