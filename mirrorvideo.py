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
    is_mov = lambda s: os.path.splitext(s)[1] == '.mov'
    join_func = lambda s: os.path.join(folder, s)
    return map(join_func, filter(is_mov, os.listdir(folder)))

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

BLANK_WINDOW = None
HIDDEN = False

def hide_window(_):
    """Hide the BLANK_WINDOW to allow user interaction on the desktop."""
    global HIDDEN
    if BLANK_WINDOW is not None:
        BLANK_WINDOW.withdraw()
        HIDDEN = True


def blank_screen():
    """Update the window overlay."""
    global BLANK_WINDOW
    if BLANK_WINDOW is None:
        try:
            BLANK_WINDOW = tk.Tk()
        except Exception as e:
            lg.warning("Exception: {}".format(e))
            return False
    BLANK_WINDOW.bind('<space>', hide_window)
    BLANK_WINDOW.bind('<Escape>', hide_window)
    if not HIDDEN:
        BLANK_WINDOW.deiconify()
        BLANK_WINDOW.configure(bg='#000')
        BLANK_WINDOW.attributes('-fullscreen', True)
        BLANK_WINDOW.wm_attributes('-topmost', True)
        BLANK_WINDOW.lift()
    return BLANK_WINDOW


def play_video(f):
    sb.call(['omxplayer', '-b', f])


def play_emotion_video(emo):
    if emo == OTHER_LABEL:
        print "OTHER_LABEL reported. That's a problem."
        return

    global mp
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
