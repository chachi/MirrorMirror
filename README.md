MirrorMirror
============

An art project.

This will become an interactive video exhibit which reacts to the
viewers facial expressions with different sorts of video actions.

Dependencies
-------------

Python: MirrorMirror is written using Python 2.7 which is installed by
default on Mac OSX, but can be found at http://python.org

opencv is currently the only library dependency. It can be installed
with a tool like homebrew from http://brew.sh or from
http://opencv.org

Running
--------

After installing OpenCV, you'll have to modify mirror.py so that the
FACE_CASCADE_XML string points to the correct
haarcascade_frontalface_default.xml file. This is necessary for the face detection.

To-Do
-------

- [ ] Add facial expression learning
- [ ] Add hooks for emotion reactions
- [ ] Distributed node communication


