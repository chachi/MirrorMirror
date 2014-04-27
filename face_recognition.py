#!/usr/bin/env python
"""
Originally found here and modified by Jack Morrison:
http://scikit-learn.org/0.11/auto_examples/applications/face_recognition.html
"""

import cv
import cv2
import os
from os.path import isfile
import string

import logging
import matplotlib.pyplot as pl
import cPickle as pickle
import re
from itertools import izip, chain

import matplotlib
matplotlib.use('MacOSX')

import numpy as np
import skimage.io as io
import skimage.exposure as exposure
from skimage.transform import resize
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


EMOTIONS = {0: "neutral",
            1: "anger",
            2: "contempt",
            3: "disgust",
            4: "fear",
            5: "happy",
            6: "sadness",
            7: "surprise"}
FACE_CASCADE_XML = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_XML = 'haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_XML)

DATA_PICKLE = 'x.pkl'
LABELS_PICKLE = 'y.pkl'
PCA_PICKLE = 'pca.pkl'
CLASSIFIER_PICKLE = 'clf.pkl'

HAPPY_LABEL = 1
OTHER_LABEL = 0
LEARN_IMG_SIZE = 64
N_COMPONENTS = 15
TARGET_NAMES = ('other', 'happy')
N_CLASSES = len(TARGET_NAMES)
H, W = (LEARN_IMG_SIZE,) * 2


def load_ck_emotion_faces():
    # Gather all emotion text files

    emotions = []
    image_files = []

    for root, dirs, files in os.walk('CK+/Emotion'):
        for txt_file in files:
            if not re.match(".*_emotion\.txt", txt_file):
                continue
            img_dir = re.sub('Emotion', 'cohn-kanade-images', root)
            img_path = os.path.join(img_dir, re.sub('_emotion\.txt',
                                                    '.png', txt_file))

            emo_file = os.path.join(root, txt_file)
            with open(emo_file, 'rU') as f:
                emotions.append(int(float(string.strip(f.readline()))))
            image_files.append(img_path)
    return emotions, image_files


def detect_and_scale_face(img):
    img_dim = img.shape
    channels = 1 if len(img_dim) < 3 else img_dim[2]

    if channels > 1:
        img = cv2.cvtColor(img, cv.CV_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if not len(faces):
        return []

    out = []
    for face_bounds in faces:
        x, y, w, h = face_bounds
        if w != h:
            print "UNSQUARE FACE DETECTED"
            os.abort()
        face_img = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(face_img, 1.1, 2)
        #if len(eyes) < 2:
            #continue

        scaled = resize(face_img, (LEARN_IMG_SIZE, LEARN_IMG_SIZE))
        out.append(scaled)

    return out


def load_data(dim):
    """We load the images into 1-D arrays because the algorithm does not
account for relative pixel positions.

    """
    emotion_files, image_files = load_ck_emotion_faces()

    happy_faces = [(i, HAPPY_LABEL)
                   for i, e in izip(image_files, emotion_files)
                   if e == 5]
    other_faces = [(i, OTHER_LABEL)
                   for i, e in izip(image_files, emotion_files)
                   if e != 5]

    n_samples = len(happy_faces) + len(other_faces)
    n_features = dim[0] * dim[1]

    X = np.empty((n_samples, n_features))
    y = np.empty((n_samples))
    idx = 0
    for fimg, emo in chain(happy_faces, other_faces):
        img = io.imread(fimg)
        faces = detect_and_scale_face(img)
        if not faces or len(faces) > 1:
            logging.info("{} faces detected in {}.".format(len(faces), fimg))
            continue
        X[idx, :] = faces[0].reshape((1, -1))
        y[idx] = emo
        idx += 1

    X.resize((idx, n_features))
    y.resize((idx))
    return X, y


def load_data_or_unpickle():
    if isfile(DATA_PICKLE) and isfile(LABELS_PICKLE):
        X = pickle.load(open(DATA_PICKLE, 'rb'))
        y = pickle.load(open(LABELS_PICKLE, 'rb'))
    else:
        X, y = load_data((H, W))
        pickle.dump(X, open(DATA_PICKLE, 'wb'))
        pickle.dump(y, open(LABELS_PICKLE, 'wb'))
    return X, y


def compute_pca(X_train, y_train):
    """Compute a PCA (eigenfaces) on the face dataset (treated as
     unlabeled dataset): unsupervised feature extraction /
     dimensionality reduction

    """
    print "Extracting the top %d eigenfaces from %d faces" % (
        N_COMPONENTS, X_train.shape[0])
    pca = RandomizedPCA(n_components=N_COMPONENTS, whiten=True).fit(X_train)
    #pca = MiniBatchSparsePCA(n_components=N_COMPONENTS).fit(X_train)
    return pca


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.Figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())
    #pl.show()


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, TARGET_NAMES, i):
    pred_name = TARGET_NAMES[int(y_pred[i])].rsplit(' ', 1)[-1]
    true_name = TARGET_NAMES[int(y_test[i])].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def plot_results(pca, X_test, y_pred, y_test):
    prediction_titles = [title(y_pred, y_test, TARGET_NAMES, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, H, W)

    # plot the gallery of the most significative eigenfaces
    eigenfaces = pca.components_.reshape((N_COMPONENTS, H, W))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, H, W, 3, 5)


def gen_classifier(X_train=None, y_train=None):
    if isfile(PCA_PICKLE):
        print "Loading PCA from file"
        pca = pickle.load(open(PCA_PICKLE, 'rb'))
    elif X_train is not None and y_train is not None:
        print "Computing PCA and saving"
        pca = compute_pca(X_train, y_train)
        pickle.dump(pca, open(PCA_PICKLE, 'wb'))
    else:
        print "Files to not exist, but no training data given"
        os.abort()

    if isfile(CLASSIFIER_PICKLE):
        print "Loading classifier from file"
        clf = pickle.load(open(CLASSIFIER_PICKLE, 'rb'))
    else:
        print "Fitting the classifier to the training set"
        param_grid = {
            'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }

        print "Projecting the input data on the eigenfaces orthonormal basis"
        X_train_pca = pca.transform(X_train)
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        print "Best estimator found by grid search:"
        print clf.best_estimator_
        pickle.dump(clf, open(CLASSIFIER_PICKLE, 'wb'))
    return pca, clf


def evaluate(pca, clf, X_test, y_test):
    """ Evaluate the results on a given test set
    """
    print "Predicting the people names on the testing set"

    X_test_pca = pca.transform(X_test)
    y_pred = clf.predict(X_test_pca)

    print 'Generating classification report'
    print classification_report(y_test, y_pred, target_names=TARGET_NAMES)

    print 'Generating confusion matrix'
    print confusion_matrix(y_test, y_pred, labels=range(N_CLASSES))
    plot_results(pca, X_test, y_pred, y_test)


def test():
    X, y = load_data_or_unpickle()
    print "Total dataset size:"
    print "n_samples: %d" % X.shape[0]
    print "n_features: %d" % X.shape[1]
    print "n_classes: %d" % N_CLASSES

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    pca, clf = gen_classifier(X_train, y_train)
    evaluate(pca, clf, X_test, y_test)


def mirror_mirror():
    pca, clf = gen_classifier()
    cam = cv2.VideoCapture(0)

    last_smile = False

    while True:
        ret, img = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
        faces = detect_and_scale_face(gray)
        if not faces:
            last_smile = False
            continue

        for scaled in faces:
            test_x = exposure.equalize_hist(scaled.reshape((1, -1)))
            face_pca = pca.transform(test_x)
            pred = clf.predict(face_pca)
            smiling = (pred == HAPPY_LABEL)
            if smiling:
                if last_smile:
                    cv2.imshow('image', scaled)
                else:
                    last_smile = True
                    break
            else:
                last_smile = False
        cv2.waitKey(1)

if __name__ == '__main__':
    mirror_mirror()
    #test()
