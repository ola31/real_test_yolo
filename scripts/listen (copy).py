#!/usr/bin/env python2
# -*- coding: utf8 -*-

#from socket import*
import socket
import time
import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
from darknet_ros_msgs.msg import BoundingBox
from std_msgs.msg import String
from sensor_msgs.msg import Image

ip = "127.0.0.1"
port = 12346

bridge=CvBridge()

def callback(data_):

    frame = bridge.imgmsg_to_cv2(data_, "bgr8")
    #cv2.imshow('roi',image)


    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()

    sock.send( str(len(stringData)).ljust(16))
    sock.send( stringData )


def image_listen():
    rospy.init_node('listen', anonymous=False)
    data = rospy.Subscriber("/camera/color/image_raw", Image, callback)
    yolo_pub=rospy.Publisher('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size = 1000)

    bridge = CvBridge()
    image = np.empty(shape=[0])
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        getdata = sock.recv(1024)  
        data2=getdata.decode("UTF-8")

        prob_s = data2.split("|")[0]
        x1_s = data2.split("|")[1]
        y1_s = data2.split("|")[2]
        x2_s = data2.split("|")[3]
        y2_s = data2.split("|")[4]
        clss_s = data2.split("|")[5]

        percent = float(prob_s)
        x1 = int(x1_s)
        y1 = int(y1_s)
        x2 = int(x2_s)
        y2 = int(y2_s)

        box = BoundingBox()
        boxes = BoundingBoxes()

        box.probability = percent
        box.xmin = x1
        box.ymin = y1
        box.xmax = x2
        box.ymax = y2
        box.Class = clss_s

        boxes.header.stamp = rospy.Time.now()
        boxes.bounding_boxes.append(box)


        yolo_pub.publish(boxes)
        rate.sleep()
    sock.close() 

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    print("connected")
    image_listen()
