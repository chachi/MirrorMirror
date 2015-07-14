import os
import subprocess as sb
import random
import Tkinter as tk
import logging as lg
from PIL import Image, ImageTk


TARGET_NAMES = ('other', 'happy', 'yawning')
OTHER_LABEL, HAPPY_LABEL, YAWNING_LABEL = range(len(TARGET_NAMES))

IMAGE_DIR = '/home/pi/images'
VIDEO_DIR = '/home/pi/videos'
SMILING_DIR = os.path.join(VIDEO_DIR, 'Smiling')
YAWNING_DIR = os.path.join(VIDEO_DIR, 'Yawning')


def list_images(folder):
    """List all the files within a directory"""
    return [os.path.join(folder, s)
            for s in os.listdir(folder)]

def list_videos(folder):
    """List all the .mov videos within a directory"""
    return [os.path.join(folder, s)
            for s in os.listdir(folder)
            if os.path.splitext(s)[1] == '.mov']

SMILING_VIDEOS = list_videos(SMILING_DIR)
YAWNING_VIDEOS = list_videos(YAWNING_DIR)
IMAGES = list_images(IMAGE_DIR)


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
    label = None

    @classmethod
    def create(cls):
        if cls.root is None:
            try:
                cls.root = tk.Tk()
                cls.root.attributes('-fullscreen', True)
                cls.root.wm_attributes('-topmost', True)
                cls.root.lift()

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

    @classmethod
    def show_image(cls):
        if cls.label is None:
            cls.label = tk.Label(cls.root)
            cls.label.pack(side='bottom', fill='both', expand='yes')

        img_path = IMAGES[random.randrange(len(IMAGES))]
        orig = Image.open(img_path)

        target_width = cls.label.winfo_width()
        target_height = cls.label.winfo_height()
        target_ratio = float(target_height) / target_width

        curr_width = orig.size[0]
        curr_height = orig.size[1]
        curr_ratio = float(curr_height) / curr_width

        # We resize the image the least amount possible to fill the
        # screen in one dimension.
        if target_ratio != curr_ratio:
            if float(curr_height) / target_height > \
               float(curr_width) / target_width:
                target_width = target_height / curr_ratio
            else:
                target_height = target_width * curr_ratio

        img = ImageTk.PhotoImage(orig.resize((int(target_width),
                                              int(target_height)),
                                             Image.BILINEAR))
        cls.label.config(image=img, bg='#000')
        cls.label.image = img
        cls.label.update()

    @classmethod
    def update(cls):
        cls.root = cls.create()
        if cls.root is not None and not cls.hidden:
            if IMAGES:
                cls.show_image()
            else:
                cls.blank()
        return cls.root


def hide_window(_):
    """Hide the OverlayWindow.root to allow user interaction on the desktop."""
    OverlayWindow.hide()

def blank_screen():
    """Update the window overlay."""
    return OverlayWindow.update()


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
