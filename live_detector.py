#!/usr/bin/python

import cPickle as pickle
import cv
import cv2
import logging
import mirrorvideo
import os
from os.path import isfile
import skimage.exposure as exposure
from skimage.transform import resize
import zmq

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


FACE_CASCADE_XML = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)

PCA_PICKLE = 'pca.pkl'
CLASSIFIER_PICKLE = 'clf.pkl'

LEARN_IMG_SIZE = 64
TARGET_NAMES = ('other', 'happy', 'yawning')
OTHER_LABEL, HAPPY_LABEL, YAWNING_LABEL = range(len(TARGET_NAMES))


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
            continue
        scaled = resize(img[y:y+h, x:x+w], (LEARN_IMG_SIZE, LEARN_IMG_SIZE))
        out.append(scaled)

    return out


def load_classifier(X_train=None, y_train=None):
    if isfile(PCA_PICKLE):
        print "Loading PCA from file"
        pca = pickle.load(open(PCA_PICKLE, 'rb'))
    else:
        print "PCA file does not exist"
        os.abort()

    if isfile(CLASSIFIER_PICKLE):
        print "Loading classifier from file"
        clf = pickle.load(open(CLASSIFIER_PICKLE, 'rb'))
    else:
        print "Class file does not exist"
        os.abort()
    return pca, clf


def classify_emotions(pca, clf, faces):
    emotions = []
    for scaled in faces:
        test_x = exposure.equalize_hist(scaled.reshape((1, -1)))
        face_pca = pca.transform(test_x)
        emotions.append(int(clf.predict(face_pca)[0]))
    return emotions


def mirror_mirror():
    pca, clf = load_classifier()
    cam = cv2.VideoCapture(0)
    cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    last_emo = OTHER_LABEL
    streak = 0
    while True:
        ret, img = cam.read()
        if not ret:
            continue

        faces = detect_and_scale_face(img)
        if not faces:
            continue

        emotions = classify_emotions(pca, clf, faces)
        if len(emotions) == 0:
            return

        matched = False
        for emo in emotions:
            if emo == OTHER_LABEL:
                continue
            elif emo == last_emo:
                matched = True
                break

        if matched:
            streak += 1
        else:
            streak = 0
            last_emo = emotions[0]

        if streak >= 4 and last_emo != OTHER_LABEL:
            mirrorvideo.play_emotion_video(last_emo)
            streak = 0
            last_emo = OTHER_LABEL

if __name__ == '__main__':
    mirror_mirror()
