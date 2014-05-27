#!/usr/bin/python

import sys
import zmq
import mirrorvideo


def connect(ctx):
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, '')
    socket.connect('tcp://{}:1776'.format(host))
    return socket


def receive(host):
    # Open ZMQ context
    ctx = zmq.Context()

    socket = connect(ctx)

    # Register callback
    while True:
        mirrorvideo.blank_screen()
        try:
            emo = socket.recv()
            print "received {}".format(emo)
            mirrorvideo.play_emotion_video(int(emo))
        except:
            socket = connect(ctx)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        host = sys.argv[1]
    else:
        host = 'mirror1.local'
    receive(host)
