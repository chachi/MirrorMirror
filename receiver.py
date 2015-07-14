#!/usr/bin/python

import sys
import zmq
import mirrorvideo
import time


def connect(ctx, host):
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, '')
    print "Connecting to {}".format(host)
    socket.connect('tcp://{}:1776'.format(host))
    return socket


def receive(host):
    while not mirrorvideo.blank_screen():
        time.sleep(10)

    # Open ZMQ context
    ctx = zmq.Context()

    print "Receiving from {}".format(host)
    class local:
        """Inner class to encapsulate nonlocal variables.  Hack around
        scoping.

        """
        socket = None
    while local.socket is None:
        try:
            local.socket = connect(ctx, host)
        except Exception as e:
            print "Failed to connect with exception {}, trying again.".format(e)
    root = mirrorvideo.blank_screen()

    def poll_events():
        window = mirrorvideo.blank_screen()
        if window:
            window.after(0, poll_events)
            try:
                emo = local.socket.recv(flags=zmq.NOBLOCK)
                print "received {}".format(emo)
                mirrorvideo.play_emotion_video(int(emo))
            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    return
                else:
                    print "Failed with {}. Retrying.".format(e)
                    local.socket = connect(ctx, host)

    root.after(0, poll_events)
    root.mainloop()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        host = sys.argv[1]
    else:
        host = 'mirror1.local'
    receive(host)
