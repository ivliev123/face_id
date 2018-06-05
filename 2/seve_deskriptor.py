
# python3 seve_deskriptor.py  -f 'da.pickle' -m 'ab'
# python3 seve_deskriptor.py  -f 'da.pickle' -m 'ab' --picamera 1

from skimage import io
from imutils.video import VideoStream
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import pickle



print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  



def index_min(array, n): #массив и номер столбца
    array_new=[]
    for i in range(len(array)):
        array_new.append(array[i][n-1])
    minimym = min(array_new)
    index=array_new.index(minimym)
    return minimym, index


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)


faceList_all=[]

faceList=[]

faceCount=0



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="использование pi камеры")
ap.add_argument("-f", "--file", type=str, default="",
	help="выбор файла")
ap.add_argument("-m", "--metod", type=str, default="ab",
	help="выбор- открытие файла для чтения, записи, дозаписи")
args = vars(ap.parse_args())



# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)


while True:
	frame = vs.read() 
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	height, width = frame.shape[:2]
	rects = detector(gray, 0)
	i=0
	if(len(rects) > 0):
		print(len(rects))
		for rect in rects:
			shape_cam = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape_cam)
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			
			if (x>0 and y>0 and x+h<width and y+h<height):
				i+=1
				start = time.time()
				face_descriptor= facerec.compute_face_descriptor(frame, shape_cam)
				print(start)
				#facedata=["none",x, y, w, h, start, face_descriptor,False,0,len(rects),"none"]
				facedata=["none",x, y, w, h, start, face_descriptor,False,0,i,"none"]
				with open(args["file"], args["metod"]) as f:
					pickle.dump(facedata, f)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        			#for (x, y) in shape2:
            			#	cv2.circle(frame, (x, y), 0, (0, 255, 255), -1)
		

    
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
