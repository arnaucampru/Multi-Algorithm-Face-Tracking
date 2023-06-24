# USAGE
# To read  video:
# python facetracking.py  --input videos/example_01.mp4
#
# To read from webcam:
# python facetracking.py

# import the necessary packages
from centroidtracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-c", "--confidence", type=float, default= -0.2,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# load HOG dlib Model
print("[INFO] loading HOG + Linear SVM people detector...")
HOGdetector = dlib.get_frontal_face_detector()

# load MIL Tracker
tracker = cv2.TrackerMIL_create()

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	#src = 0 laptop camera src = 1 external webcam
	vs = VideoStream(src=0).start() 
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])


# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our trackers, followed by a dictionary to
# map each unique object ID
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []

# initialize the total number of frames processed 
totalFrames = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]


	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []
	
	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		# convert the frame to a boxesand obtain the detections
		
		#HOG Face detector
		boxes,weights,idx = HOGdetector.run(rgb, 1,-1)



		# loop over the detections
	
		for i,box in enumerate(boxes):
			if weights[i] > args["confidence"]:
				startX = box.left()
				startY = box.top()
				endX =  box.right()
				endY = box.bottom()
				wide = endX - startX
				high = endY - startY
				
				# draw the detected face
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(255, 0, 0), 2)
				
				# construct rectangle object from the bounding
				# box coordinates and then start the  correlation
				# tracker
				bbox = (startX, startY, wide, high)
				ok = tracker.init(rgb, bbox)

				# add the tracker to our list of trackers so we can
				# utilize it during skip frames
				rects.append((startX, startY, endX, endY))
				trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			ok,pos = tracker.update(rgb)
			#pos = tracker.get_position()

			# unpack the position object
			startX = int(pos[0])
			startY = int(pos[1])
			wide = int(pos[2])
			high = int(pos[3])
			endX = startX + wide
			endY = startY + high
			
			

			# add the bounding box coordinates to the rectangles list
			rects.append((startX,startY,endX,endY))


	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	
	# status of the cicle loop
	text = "{}: {}".format("Status", status)
	cv2.putText(frame, text, (10, H - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	


	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
