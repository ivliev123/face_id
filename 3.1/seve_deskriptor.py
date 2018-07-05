
# python3 seve_deskriptor.py 
# python3 seve_deskriptor.py  --picamera 1

from skimage import io
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import datetime
import argparse
import imutils
import dlib
import time
import cv2
import pickle



print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="использование pi камеры")

args = vars(ap.parse_args())


vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

start = time.time()
while True:
	frame = vs.read() 
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	delta_time=time.time()-start
	if delta_time>0.5 and len(rects) > 0:

		start = time.time()
		cv2.imwrite('info/'+str(start)+'.jpg',frame)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
