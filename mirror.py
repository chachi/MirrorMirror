#!/usr/bin/env python
import cv2
import cv

FACE_CASCADE_XML = 'haarcascade_frontalface_default.xml'


def mirror_mirror():
    # Load the face detector details
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)

    # Start the camera
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('mirror', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    mirror_mirror()
