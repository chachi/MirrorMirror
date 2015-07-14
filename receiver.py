#!/usr/bin/python

import os
import sys
import zmq
import mirrorvideo
import time
from datetime import datetime
from dotenv import load_dotenv
import logging as lg

lg.basicConfig(level=lg.INFO, format='%(asctime)s %(message)s')

IMAGE_TIMEOUT = 1               # Seconds


def connect(ctx, host):
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, '')
    lg.info("Connecting to {}".format(host))
    socket.connect('tcp://{}:1776'.format(host))
    return socket


def receive(host):
    while not mirrorvideo.blank_screen():
        time.sleep(10)

    # Open ZMQ context
    ctx = zmq.Context()

    lg.info("Receiving from {}".format(host))
    class local:
        """Inner class to encapsulate nonlocal variables.  Hack around
        scoping.

        """
        socket = None
        last_update = datetime.min

    while local.socket is None:
        try:
            local.socket = connect(ctx, host)
        except Exception as e:
            lg.info(
                "Failed to connect with exception {}, trying again.".format(e))
    root = mirrorvideo.blank_screen()

    def poll_events():
        window = mirrorvideo.OverlayWindow.create()

        now = datetime.now()
        if (now - local.last_update).total_seconds() > IMAGE_TIMEOUT:
            mirrorvideo.blank_screen()
            local.last_update = now
        if window:
            window.after(0, poll_events)
            try:
                emo = local.socket.recv(flags=zmq.NOBLOCK)
                lg.info("received {}".format(emo))
                mirrorvideo.play_emotion_video(int(emo))
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    return
                else:
                    lg.info("Failed with {}. Retrying.".format(e))
                    local.socket = connect(ctx, host)

    root.after(0, poll_events)
    root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-detector',
                        default='mirror1.local', action='store')
    parser.add_argument('-env',
                        default='/home/pi/Desktop/config.txt',
                        action='store')
    args = parser.parse_args()

    load_dotenv(args.env)
    global IMAGE_TIMEOUT
    IMAGE_TIMEOUT = float(os.environ.get('IMAGE_TIMEOUT', 1))
    mirrorvideo.MediaRepository.init()
    receive(args.detector)


if __name__ == '__main__':
    main()
