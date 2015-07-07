#!/usr/bin/python

import sys
import zmq
import mirrorvideo


def connect(ctx, host):
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, '')
    print "Connecting to {}".format(host)
    socket.connect('tcp://{}:1776'.format(host))
    return socket


def receive(host):
    # Open ZMQ context
    ctx = zmq.Context()

    print "Receiving from {}".format(host)

    socket = None
    while socket is None:
        try:
            socket = connect(ctx, host)
        except Exception as e:
            print "Failed to connect with exception {}, trying again.".format(e)

    # Register callback
    while True:
        mirrorvideo.blank_screen()
        try:
            print "Listening to socket"
            emo = socket.recv()
            print "received {}".format(emo)
            mirrorvideo.play_emotion_video(int(emo))
        except Exception as e:
            print "Failed with {}. Retrying.".format(e)
            socket = connect(ctx, host)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        host = sys.argv[1]
    else:
        host = 'mirror1.local'
    receive(host)
