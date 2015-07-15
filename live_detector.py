#!/usr/bin/python

import cPickle as pickle
import cv
import cv2
import logging as lg
import mirrorvideo
import os
from os.path import isfile
import skimage.exposure as exposure
from skimage.transform import resize
import zmq
import time
import subprocess as sb
from dotenv import load_dotenv


# Display progress logs on stdout
lg.basicConfig(level=lg.INFO,
               format='%(asctime)s [%(levelname)-8s] %(message)s')

FACE_CASCADE_XML = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)

PCA_PICKLE = 'pca.pkl'
CLASSIFIER_PICKLE = 'clf.pkl'

LEARN_IMG_SIZE = 64
VIDEO_PAUSE = 10.0


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
            lg.error("UNSQUARE FACE DETECTED")
            continue
        scaled = resize(img[y:y+h, x:x+w], (LEARN_IMG_SIZE, LEARN_IMG_SIZE))
        out.append(scaled)

    return out


def load_classifier(X_train=None, y_train=None):
    if isfile(PCA_PICKLE):
        lg.info("Loading PCA from file")
        pca = pickle.load(open(PCA_PICKLE, 'rb'))
    else:
        lg.error("PCA file does not exist")
        os.abort()

    if isfile(CLASSIFIER_PICKLE):
        lg.info("Loading classifier from file")
        clf = pickle.load(open(CLASSIFIER_PICKLE, 'rb'))
    else:
        lg.error("Class file does not exist")
        os.abort()
    return pca, clf


def classify_emotions(pca, clf, faces):
    emotions = []
    for scaled in faces:
        test_x = exposure.equalize_hist(scaled.reshape((1, -1)))
        face_pca = pca.transform(test_x)
        emotions.append(int(clf.predict(face_pca)[0]))
    return emotions


def clear_buffer(cam):
    for i in xrange(7):
        cam.grab()


def mirror_mirror():
    lg.info("mirror_mirror")
    pca, clf = load_classifier()
    for i in xrange(6):
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            break
        lg.info("Cam not open. Sleeping.")
        time.sleep(10)
    if not cam.isOpened():
        sb.call(['sudo', 'reboot'])

    lg.info("Camera is open.")
    cam.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind('tcp://*:1776')

    last_emo = mirrorvideo.OTHER_LABEL
    streak = 0
    lg.info("Starting captures..")
    while True:
        ret, img = cam.read()
        if not ret:
            continue
        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)

        faces = detect_and_scale_face(img)
        if not faces:
            clear_buffer(cam)
            continue

        emotions = classify_emotions(pca, clf, faces)
        if len(emotions) == 0:
            return

        matched = False
        for emo in emotions:
            if emo == mirrorvideo.OTHER_LABEL:
                continue
            elif emo == last_emo:
                matched = True
                break

        if matched:
            streak += 1
        else:
            streak = 0
            last_emo = emotions[0]

        if streak >= 2 and last_emo != mirrorvideo.OTHER_LABEL:
            lg.info("Saw a {} face".format(str(emo)))
            socket.send(str(emo))
            streak = 0
            last_emo = mirrorvideo.OTHER_LABEL
            clear_buffer(cam)
            time.sleep(float(VIDEO_PAUSE))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-env',
                        default='/home/pi/Desktop/config.txt',
                        action='store')
    args = parser.parse_args()
    load_dotenv(args.env)

    VIDEO_PAUSE = os.environ.get('VIDEO_PAUSE', VIDEO_PAUSE)
    mirror_mirror()
