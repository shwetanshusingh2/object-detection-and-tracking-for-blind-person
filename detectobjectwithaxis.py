import numpy as np
import time
import cv2
import random
import os
from gtts import gTTS 
import pygame
import pyttsx3
language = 'en'

con = 0.5
thresh = 0.3
# load the COCO class labels our YOLO model was trained on
# read class names from text file
LABELS = None
with open('coco.names', 'r') as f:    ########## define path to YOLO classes text file
    LABELS = [line.strip() for line in f.readlines()]

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")


# load our YOLO object detector trained on COCO dataset (80 classes)

net = cv2.dnn.readNet('yolov2.weights', 'yolov2.cfg')
engine = pyttsx3.init()

def process():
	vs = cv2.VideoCapture(0)
	oldvr=[]
	while True:
		vr=[]
		ret,image = vs.read()
		#frame = imutils.resize(frame, width=600)
		# load our input image and grab its spatial dimensions
		#image = cv2.imread('dog.jpg')  #### define path to desired image
		(H, W) = image.shape[:2]
	
		# determine only the *output* layer names that we need from YOLO
		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	
		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# show timing information on YOLO
		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
	
		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > con:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					
					box = detection[0:4] * np.array([W, H, W, H])
					print(box)
					(centerX, centerY, width, height) = box.astype("int")
	
					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		
		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, con, thresh)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
	
				# draw a bounding box rectangle and label on the image
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
				print("old vector")
				print(oldvr)
				labeld=LABELS[classIDs[i]]
				if labeld in oldvr:
					for j in range(0,len(oldvr)):
						if oldvr[j] == labeld:
							print("label availabe",oldvr[j],labeld)
							print("X axis",oldvr[j+1],"Now x value",x)
							d=oldvr[j+1]-x
							print("d value",d)
							if d >0.0:
								print(labeld,"Moving to right direction from["+str(oldvr[j+1])+","+str(oldvr[j+2])+"]to["+str(x)+","+str(y)+"]")
								msg=labeld+"Moving to right direction from"+str(oldvr[j+1])+","+str(oldvr[j+2])+"to"+str(x)+","+str(y)
								engine.say(msg)
								engine.runAndWait()
								msg=""
														
							if d <0.0:
								print(labeld,"Moving to left direction from["+str(oldvr[j+1])+","+str(oldvr[j+2])+"]to["+str(x)+","+str(y)+"]")
								msg=labeld+"Moving to left direction from"+str(oldvr[j+1])+","+str(oldvr[j+2])+"to"+str(x)+","+str(y)
								engine.say(msg)
								engine.runAndWait()
								msg=""
			
				vr.append(LABELS[classIDs[i]])
				vr.append(x)
				vr.append(y)
		print(vr)
		oldvr=vr
		cv2.imshow("Surveillance Camera", image)
		key = cv2.waitKey(1) & 0xFF

		#if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
#
	# do a bit of cleanup
	cv2.destroyAllWindows()
process()
