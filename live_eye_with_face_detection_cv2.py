#!/usr/bin/python3
from flask import request
import cv2
import base64
from twilio.rest import Client
from msg import message
import numpy as np
# loading  face trained  data

facehaar=cv2.CascadeClassifier('face.xml')
eyehaar=cv2.CascadeClassifier('eye.xml')
#  starting  camera
def face():
    cap=cv2.VideoCapture(0)
    status,image = cap.read()
    image = cv2.flip(image, 1)
    #  face detector  apply in  virat_img--scalling  range 
    face_only=facehaar.detectMultiScale(image,1.15,5)
	#print(face_only)
    for  x,y,w,h in  face_only:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        face_image = image[y:y+h, x:x+w]
        eye=eyehaar.detectMultiScale(face_image)
        for x,y,w,h in eye:
            cv2.rectangle(face_image,(x,y),(x+w,y+h),(250,250,0),2)
                           
    if image.nonzero():
        message()
        exit()
