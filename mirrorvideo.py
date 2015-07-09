import os
import subprocess as sb
import time
import random
import Tkinter as tk
import logging as lg

TARGET_NAMES = ('other', 'happy', 'yawning')
OTHER_LABEL, HAPPY_LABEL, YAWNING_LABEL = range(len(TARGET_NAMES))

video_dir = '/home/pi/videos'
smiling_dir = os.path.join(video_dir, 'Smiling')
yawning_dir = os.path.join(video_dir, 'Yawning')


def list_videos(folder):
    is_mov = lambda s: os.path.splitext(s)[1] == '.mov'
    join_func = lambda s: os.path.join(folder, s)
    return map(join_func, filter(is_mov, os.listdir(folder)))

smiling_videos = list_videos(smiling_dir)
yawning_videos = list_videos(yawning_dir)


def get_video(emo):
    if emo == HAPPY_LABEL:
        videos = smiling_videos
    else:
        videos = yawning_videos
    if not videos:
        return ''
    return videos[random.randrange(len(videos))]

BLANK_WINDOW = None

def blank_screen():
    global BLANK_WINDOW
    if BLANK_WINDOW is None:
        try:
            BLANK_WINDOW = tk.Tk()
        except Exception as e:
            lg.warning("Exception: {}".format(e))
            return False
    BLANK_WINDOW.deiconify()
    BLANK_WINDOW.configure(bg='#000')
    BLANK_WINDOW.attributes('-fullscreen', True)
    BLANK_WINDOW.wm_attributes('-topmost', True)
    BLANK_WINDOW.lift()
    return True


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
    print "Video playing, sleeping."
    time.sleep(30)
    print "Ready again."
