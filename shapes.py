import cv2
import imutils
import numpy

# reading image
img = cv2.imread("shapes.png")
cv2.imshow("original image",img)
cv2.waitKey(0)

#image conversions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_inv = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]	
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]	
adaptiveThreshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,231,1)

#image contours
cnts = cv2.findContours(adaptiveThreshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = img.copy()

# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (0, 0, 255), 3)
	cv2.imshow("Shape Detection", output)
	cv2.waitKey(0)

	
