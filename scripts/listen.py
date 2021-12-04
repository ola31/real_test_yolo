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
port = 12345

bridge=CvBridge()

def callback(data_):

    frame = bridge.imgmsg_to_cv2(data_, "bgr8")
    #print(frame.shape[0])
    #print(frame.shape[1])
    #cv2.imshow('roi',image)
    #image_toresize = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_AREA)
 

    #height = 360
    #width = 640
    #blank_image = np.zeros((480, 640, 3), np.uint8)
   # blank_image[:,:] = (255,255,255)
   # l_img = blank_image.copy()
   # x_offset = 0
   # y_offset = 60
   # l_img[y_offset:y_offset+height, x_offset:x_offset+width] = image_toresize.copy()
   
    

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()

    #String 형태로 변환한 이미지를 socket을 통해서 전송
    sock.send( str(len(stringData)).ljust(16))
    sock.send( stringData )
    #sock.close()

#다시 이미지로 디코딩해서 화면에 출력. 그리고 종료
    #decimg=cv2.imdecode(data,1)
    #cv2.imshow('CLIENT',decimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def image_listen():
    rospy.init_node('listen', anonymous=False)
    data = rospy.Subscriber("/camera/color/image_raw", Image, callback,queue_size = 1)
    yolo_pub=rospy.Publisher('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size = 1)

    bridge = CvBridge()
    image = np.empty(shape=[0])
  #  yolo_pub=rospy.Publisher('/darknet_ros/bounding_boxes', BoundingBoxes, queue_size = 1000)
  # chatt_pub=rospy.Publisher("/chatt",String, queue_size = 1000)

  #  yolo_data=BoundingBoxes()
  #  chatt_msg=String()
    #rospy.spin()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        #clientSocket.send("I am a client".encode("UTF-8"))
        #print("send_massage.")


        getdata = sock.recv(1024)  #데이터 수신
        #print("recieved data :"+data.decode("UTF-8"))
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

        if(x1 < 0):
            x1 = 0
        if(x2 < 0):
            x2 = 0
        if(y1 < 0):
            y1 = 0
        if(y2 < 0):
            y2 = 0
        if(x1 > 1280):
            x1=1280
        if(y1 > 720):
            y1=720
        if(x2 > 1280):
            x2=1280
        if(y2 > 720):
            y2=720
        
        


        #print(x1)
        #print(y1)88
        #print(x2)
        #print(y2)

        box = BoundingBox()
        boxes = BoundingBoxes()

        box.probability = percent
        x1 = int((x2 - x1)*0.3 + x1)
        x2 = int(x2 - (x2 - x1)*0.3) 
        y1 = int((y2 - y1)*0.3 + y1)
        y2 = int(y2 - (y2 - y1)*0.3)
        box.xmin = x1
        box.ymin = y1
        box.xmax = x2
        box.ymax = y2
        box.Class = clss_s

        boxes.header.stamp = rospy.Time.now()
        boxes.bounding_boxes.append(box)


        yolo_pub.publish(boxes)

        '''yolo_data.header.stamp=rospy.Time.now()
        yolo_data.header.frame_id ="detection"
        yolo_data.image_header.frame_id ="detection"
        yolo_data.bounding_boxes[10].probability = 0.7
        yolo_data.bounding_boxes[0].xmin = 301
        yolo_data.bounding_boxes[0].xmax = 171
        yolo_data.bounding_boxes[0].ymin = 555
        yolo_data.bounding_boxes[0].ymax = 480
        yolo_data.bounding_boxes[0].id = 0
        yolo_data.bounding_boxes[0].Class = "person"

        yolo_pub.publish(yolo_data)
        chatt_msg.data=data2
        chatt_pub.publish(chatt_msg)'''
        rate.sleep()
    sock.close()  # 연결 종료



    #rospy.spin()



if __name__ == '__main__':

   # clientSocket = socket(AF_INET,SOCK_STREAM)  #소켓 생성
   # clientSocket.connect((ip,port))   # 서버와 연결
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    print("connected")
    image_listen()
