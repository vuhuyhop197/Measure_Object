# USAGE
# python M2_newbrach.py -w (reference object dimention the bigger)

# import the necessary packages
import cv2
from scipy.spatial import distance as distance
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import urllib 
from urllib import request

i1x,i1y = -1,-1
i2x,i2y = -1,-1
calib_flag = False
state=0

def nothing(x):
	pass
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# mouse callback function
def get_point(event,x,y,flags,param):
	global i1x,i1y,i2x,i2y,state,calib_flag
	if event == cv2.EVENT_LBUTTONDBLCLK:
		state = state +1 
		if state == 1:
			i1x,i1y = x,y
		if (state == 2):
			i2x,i2y = x,y
		if state == 3:
			state = 0
			i1x = -1
			i2x = -1

			 
# construct the argument parse and parse the arguments
def calib_object(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	candyaa = cv2.getTrackbarPos('candya', 'trackbar')
	candybb = cv2.getTrackbarPos('candyb', 'trackbar')
	edged = cv2.Canny(gray, candyaa, candybb)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	cv2.imshow('egde',edged)
	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)
	# loop over the contours individually
	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 500:
			continue
		# compute the rotated bounding box of the contour
		orig = image.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		# draw the midpoints on the image


		# compute the Euclidean distance between the midpoints
		dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		global pixelsPerMetric
		pixelsPerMetric = dA / args["width"]
		if(dB >dA):
			pixelsPerMetric = dB / args["width"]
		calib_flag = True
		#return parameter for display
		return tltrX, tltrY,blbrX, blbrY,tlblX, tlblY,trbrX, trbrY,calib_flag
		

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in cm)")
args = vars(ap.parse_args())
# imgResp=urllib.request.urlopen(url)
  ## img=cv2.imdecode(imgNp,-1)
#ipcam
#url='http://192.168.1.29:8080/shot.jpg'

#while True:
 #   imgResp=urllib.request.urlopen(url)
  ## img=cv2.imdecode(imgNp,-1)

    # all the opencv processing is done here
   
    #if ord('q')==cv2.waitKey(10):
     #   exit(0)
#END_ipcam
#cap= cv2.VideoCapture('http://admin:12345678@192.168.1.12:8080/video')
cap= cv2.VideoCapture(1)
cv2.namedWindow('trackbar')
cv2.createTrackbar('candya','trackbar',200,255,nothing)
cv2.createTrackbar('candyb','trackbar',50,255,nothing)
pixelsPerMetric = 0.0
tlblX, tlblY,trbrX, trbrY=0.0,0.0,0.0,0.0 
cv2.namedWindow('Image')
cv2.setMouseCallback('Image',get_point)
with np.load('outfile.npz') as X:
    mtx, dist= [X[i] for i in ('mtx','dist')]
#Load parameter of camera
while True:
	ret, image = cap.read()
	h, w = image.shape[:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	
# # undistort
	image = cv2.undistort(image, mtx, dist, None, newcameramtx)

# # crop the image
	x,y,w,h = roi
	image = image[y:y+h, x:x+w]
	#compute the size of the object
	if calib_flag == True:
		dA = distance.euclidean((i1x, i1y), (i2x, i2y))
		dimA = dA / pixelsPerMetric
		# print(dimA)
		cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)
		cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
		# draw lines between the midpoints
		if state==2:
			cv2.line(image, (int(i1x), int(i1y)), (int(i2x), int(i2y)), (255, 0, 255), 2)

		# draw the object sizes on the image
			cv2.putText(image, "{:.1f}cm".format(dimA),
				(int((i1x+i2x)/2 - 15), int((i1y+i2y)/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
		cv2.circle(image,(i1x,i1y),2,(255,0,0),-1)
		cv2.circle(image,(i2x,i2y),2,(255,0,0),-1)

		# show the output image
	
	if cv2.waitKey(10) & 0xff == ord('q'):
		break
	if cv2.waitKey(10) & 0xff == ord('c'):
		ret, image = cap.read()
		tltrX, tltrY,blbrX, blbrY,tlblX, tlblY,trbrX, trbrY,calib_flag=calib_object(image)
		print(tlblX, tlblY,trbrX, trbrY)
	if cv2.waitKey(10) & 0xff == ord('r'):
		i1x,i1y = -1,-1
		i2x,i2y = -1,-1
	cv2.imshow("Image", image)
cv2.destroyAllWindows()
