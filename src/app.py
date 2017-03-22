'''
Created on Aug 10, 2015
@author: mattjarvis

__ OVERVIEW __
A class that offers a global reference to primary functions & data.
'''
import time
import eventlet

IP = '192.168.1.75'
PORT = 8080
LOG_TAG_LENGTH = 15
LOG_LINE_LENGTH = 50


def get_ip():
    return IP


def get_port():
    return PORT


def sleep(seconds=0):
    eventlet.sleep(seconds)


def log(tag, *msgs):
    tag_padding = ' ' * (LOG_TAG_LENGTH - (len(tag) % LOG_TAG_LENGTH))
    padded_tag = tag_padding + tag
    msg = ''.join((' ' + str(m) for m in msgs))
    msg_padding = '.' * (LOG_LINE_LENGTH - (len(msg) % LOG_LINE_LENGTH))
    padded_msg = msg + '. ' + msg_padding
    timestamp = time.strftime("%I:%M:%S%p on %d-%b-%Y")
    print padded_tag + ':', padded_msg, timestamp
