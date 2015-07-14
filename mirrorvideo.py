import os
import subprocess as sb
import time
import random
import Tkinter as tk
import logging as lg

TARGET_NAMES = ('other', 'happy', 'yawning')
OTHER_LABEL, HAPPY_LABEL, YAWNING_LABEL = range(len(TARGET_NAMES))

VIDEO_DIR = '/home/pi/videos'
SMILING_DIR = os.path.join(VIDEO_DIR, 'Smiling')
YAWNING_DIR = os.path.join(VIDEO_DIR, 'Yawning')


def list_videos(folder):
    """List all the .mov videos within a directory"""
    return [os.path.join(folder, s)
            for s in os.listdir(folder)
            if os.path.splitext(s)[1] == '.mov']

SMILING_VIDEOS = list_videos(SMILING_DIR)
YAWNING_VIDEOS = list_videos(YAWNING_DIR)


def get_video(emo):
    """Get a random video for the given emotion."""
    if emo == HAPPY_LABEL:
        videos = SMILING_VIDEOS
    else:
        videos = YAWNING_VIDEOS
    if not videos:
        return ''
    return videos[random.randrange(len(videos))]


class OverlayWindow(object):
    root = None
    hidden = False

    @classmethod
    def create(cls):
        if cls.root is None:
            try:
                cls.root = tk.Tk()
                cls.root.bind('<space>', hide_window)
                cls.root.bind('<Escape>', hide_window)
            except Exception as e:
                lg.warning("Exception: {}".format(e))
                return False
        return cls.root

    @classmethod
    def blank(cls):
        if cls.root and not cls.hidden:
            cls.root.deiconify()
            cls.root.configure(bg='#000')
            cls.root.attributes('-fullscreen', True)
            cls.root.wm_attributes('-topmost', True)
            cls.root.lift()

    @classmethod
    def hide(cls):
        if cls.root is not None:
            cls.root.withdraw()
            cls.hidden = True

def hide_window(_):
    """Hide the OverlayWindow.root to allow user interaction on the desktop."""
    OverlayWindow.hide()

def blank_screen():
    """Update the window overlay."""
    root = OverlayWindow.create()
    OverlayWindow.blank()
    return root


def play_video(f):
    #sb.call(['omxplayer', '-b', f])
    pass


def play_emotion_video(emo):
    if emo == OTHER_LABEL:
        print "OTHER_LABEL reported. That's a problem."
        return
    success = False
    fname = get_video(emo)
    while not success:
        try:
            blank_screen()
            play_video(fname)
            blank_screen()
            success = True
        except:
            pass
    print "Ready again."
