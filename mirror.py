#!/usr/bin/env python
import cv2
import cv

FACE_CASCADE_XML = 'haarcascade_frontalface_default.xml'


def show_face(face_bounds, img):
    x, y, w, h = face_bounds
    my_face = img[y:y+h, x:x+w, :]
    if my_face.shape[0] != 0:
        cv2.imshow('mirror', my_face)


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
        if len(faces) == 0:
            continue

        for face in faces:
            show_face(face, img)
        cv2.waitKey(1)

if __name__ == '__main__':
    mirror_mirror()
