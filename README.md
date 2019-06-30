# Save Electricity using Machine Learning
![Detection of light using openCV](https://user-images.githubusercontent.com/40906718/60383821-d8391380-9a93-11e9-8db0-48206d1f9f18.jpg)
From this we can find the status of the light, If the lights are 'ON' then the camera checks for presence of Human in the room if there is no Human Presence then it will send the text message to owner of house.
Built using [openCV](http://opencv.net/)'s state-of-the-art camera built with machine learning.
[![PyPI]](https://pypi.org/)
[![Documentation Status](https://readthedocs.org/projects/light-detection/badge/?version=latest)](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)

## Features

#### Find status of Light on Live Camera

Find all the status of light from live camera:

'''
There are four different packages and you should select only one of them. Do not install multiple different packages in the same environment. There is no plugin architecture: all the packages use the same namespace (cv2). If you installed multiple different packages in the same environment, uninstall them all with pip uninstall and reinstall only one package.

a. Packages for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

    run pip install opencv-python if you need only main modules
    run pip install opencv-contrib-python if you need both main and contrib modules (check extra modules listing from OpenCV documentation)

b. Packages for server (headless) environments

These packages do not contain any GUI functionality. They are smaller and suitable for more restricted environments.

    run pip install opencv-python-headless if you need only main modules
    run pip install opencv-contrib-python-headless if you need both main and contrib modules (check extra modules listing from OpenCV documentation)

Import the package:

import cv2

All packages contain haarcascade files. cv2.data.haarcascades can be used as a shortcut to the data folder. For example:

cv2.CascadeClassifier(cv2.data.haarcascades + "face.xml")
'''
#### Find and check the status of light

Get the locations and outlines of each person's eyes, nose, mouth and chin.

```python
from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
```
#### Identify status of Light

Recognize what is status.

```python
# load the image, convert it to grayscale, and blur it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
```

You can even use this library with other Python libraries :

![](https://user-images.githubusercontent.com/40906718/60395083-b8fdbd00-9b4b-11e9-8cce-8cd6ad4b1f3c.png)

#### Detect the Human Presence
For this we need only two things :
* OpenCV
* [Download the face datasets](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalcatface.xml)
* [Download the eyes dtatests](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml)

Detect the Face and Eyes of Human
![unknown](https://opencv-python-tutroals.readthedocs.io/en/latest/_images/face.jpg)
```python
import cv2
import numpy as np
# loading  face trained  data
facehaar=cv2.CascadeClassifier('face.xml')
eyehaar=cv2.CascadeClassifier('eye.xml')
```
#### Sending Text Message
For sending text message we need to download Twilio API:
  * [Download twilio API](https://pypi.org/project/twilio/)
Genrally Twilio Api used for sending text message, Whatsapp message and we can use it for call using python language.
![](https://s3.amazonaws.com/com.twilio.prod.twilio-docs/original_images/sms-2-way.png)

```python
from twilio.rest import Client
def message():
# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
    account_sid = '# your account_sid'
    auth_token = '# your auth token'
    client = Client(account_sid, auth_token)
```
You can free create Account on Twilio from which you get your account_sid and auth_token which we can use to send text messageas well as whatsapp message.

## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have OpenCV already installed with Python bindings:

  * [How to install OpenCV from source on macOS or Ubuntu](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install opencv-contrib-python
```

#### Installing on Windows

While Windows isn't officially supported, helpful users have posted instructions on how to install this library:

  * [@masoudr's Windows 10 installation guide (openCV)](https://docs.opencv.org/3.1.0/d5/de5/tutorial_py_setup_in_windows.html)

#### Installing a more required python libraries
  * [Download the imutils](https://pypi.org/project/imutils/).
  * [Download the Scikit-image](https://scikit-image.org/).
  * [Download the numpy](https://www.numpy.org/) 

# How TO START (Working)
First of all You need to install FLASK for using it on your server:
  * [Flask](http://flask.pocoo.org/docs/1.0/installation/)
  * [Flask](https://pypi.org/project/Flask/)
  
Then you has to just clone this Repository for further working.
Now you have to run the FLASK server for `main.py` file and Copy the link to browser.
Then the you have to just go to to the first page of the site which is Login or Registeration Page.
If you are existing user then just do login else you have to register.
Then you came to the Profile of the page.
After you have to change the link from "localhost:port" ---> "localhost:port/camera" then the working of the program starts.

#### First Detection Of Light
After importing useful libraries which was declared above:

```python
# load the image, convert it to grayscale, and blur it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
```

The output of these operations can be seen below:

![](https://user-images.githubusercontent.com/40906718/60395083-b8fdbd00-9b4b-11e9-8cce-8cd6ad4b1f3c.png)

```python
# threshold the image to reveal light regions in the
# blurred image
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
```
This operation takes any pixel value p >= 200 and sets it to 255 (white). Pixel values < 200 are set to 0 (black).
After thresholding we are left with the following image:

![](https://user-images.githubusercontent.com/40906718/60395917-20216e80-9b58-11e9-9d52-114180d08e94.png)

Note how the bright areas of the image are now all white while the rest of the image is set to black.
However, there is a bit of noise in this image (i.e., small blobs), so let’s clean it up by performing a series of erosions and dilations:
```python
# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
```
After applying these operations you can see that our thresh  image is much “cleaner”, although we do still have a few left over blobs that we’d like to exclude (we’ll handle that in our next step):

![](https://user-images.githubusercontent.com/40906718/60395937-873f2300-9b58-11e9-9cba-a2e3a706484f.png)

The critical step in this project is to label each of the regions in the above figure; however, even after applying our erosions and dilations we’d still like to filter out any leftover “noisy” regions.
An excellent way to do this is to perform a connected-component analysis:

```python
# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large"
# components
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(thresh.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 300:
		mask = cv2.add(mask, labelMask)
```

I have provided a GIF animation below that visualizes the construction of the labelMask  for each label . Use this animation to help yourself understand how each of the individual components are accessed and displayed:

![known](https://user-images.githubusercontent.com/40906718/60395274-adf85c00-9b4e-11e9-830c-3412ffb98fa0.gif)

Then counts the number of non-zero pixels in the labelMask . If numPixels  exceeds a pre-defined threshold (in this case, a total of 300 pixels), then we consider the blob “large enough” and add it to our mask .
The output mask  can be seen below:

![](https://user-images.githubusercontent.com/40906718/60395985-1fd5a300-9b59-11e9-94fb-3839028f2ef3.png)

You should then see the following output image:

![](https://user-images.githubusercontent.com/40906718/60383845-336b0600-9a94-11e9-954d-0e1c59e18873.png)

#### Detection of Human Presence
In this we try detect the face of a Human in room.
For running this we need libraries which shown above and both the datasets of the face as well as eyes.

```python
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
```
From this we will get a live camera in which the whole face detected with comes inside the red rectangle and both the eyes shown inside the green rectangle.
Here we uses the Cascade Classifier with openCV for detecting face.
Here is the image shown below:

![](https://www.stoltzmaniac.com/wp-content/uploads/2017/06/modifiedWebcamShot-smaller-min-620x349.png)

#### Now the Message part

In this we uses the Twilio python API for sending message to the owner of the house.
Use the Twilio API for SMS to send and receive text messages over the carrier network to any phone, anywhere in the world. From a Twilio phone number, SMS messages are delivered globally.
 
![](https://s3.amazonaws.com/com.twilio.prod.twilio-docs/images/addons.width-800.png)

```python
from twilio.rest import Client

def message():
# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
    account_sid = '# your account_sid'
    auth_token = '# your auth_token'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                     body=" **** YOU LEFT A LIGHT ON **** ",     # message you wanted to send
                     from_='# your twilio number ',              # thhis number you have to choose for sending message 
                     to='# senders number'                        # you have to verify the number first on which you want to send the message
                                    )

    print(message.sid)                 # print the confirmation of message have been send
```

From all this things you will bwe avle to run the message if the lights are 'ON; but there is no HUman in the room.

## Important Points:

when you clone the Repository :
/*Save_Electricity_using_ML
   /*static
       /*style
   /*templates
       /*home
       /*index
       /*layout
       /*profile
       /*register
   /*detect_bright_spots.py
   /*eye.xml
   /*face.xml
   /*live_eye_with_face_detection_cv2.py
   /*main.py
