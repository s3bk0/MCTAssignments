import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.model_selection  import train_test_split, KFold, cross_val_score

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, 
        hspace=0.05, wspace=0.05)

    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(1, top_n, i + 1, xticks=[], 
            yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))

def evaluate_cross_val(clf, X, y, K):
    cv = KFold(n_splits=K, shuffle=True)

    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), np.std(scores, ddof=1)/np.sqrt(K)))

def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)

    print("Accuracy on training set: ", clf.score(x_train, y_train))
    print("Accuracy on test set: ", clf.score(x_test, y_test))

    ypred = clf.predict(x_test)

    print(sk.metrics.classification_report(y_test, ypred))
    print(sk.metrics.confusion_matrix(y_test, ypred))

faces = fetch_olivetti_faces()

randmask = np.random.rand(*faces["target"].shape) < 1# 0.1
# print_faces(faces["images"][randmask], faces["target"][randmask], 20)

svc_1 = SVC(kernel='linear')

x_train, x_test, y_train, y_test = train_test_split(faces['data'], faces['target'], test_size=0.25) 

evaluate_cross_val(svc_1, x_train, y_train, 5)
train_and_evaluate(svc_1, x_train, x_test, y_train, y_test)

#people wearing glasses
def create_target(segments):
    out = np.zeros(faces["target"].shape[0])
    for (start, end) in segments:
        out[start:end] = 1
    return out

glasses = [
   (10, 19), (30, 32), (37, 38), (50, 59), (63, 64),
   (69, 69), (120, 121), (124, 129), (130, 139), (160, 161),
   (164, 169), (180, 182), (185, 185), (189, 189), (190, 192),
   (194, 194), (196, 199), (260, 269), (270, 279), (300, 309),
   (330, 339), (358, 359), (360, 369)
]

target_glasses = create_target(glasses)

X_train, X_test, Y_train, Y_test = train_test_split(faces['data'], target_glasses)

svc2 = SVC(kernel='linear')

evaluate_cross_val(svc2, X_train, Y_train, 5)
train_and_evaluate(svc2, X_train, X_test, Y_train, Y_test)



plt.show()