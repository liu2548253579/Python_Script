import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
w = 9   # 10 - 1
h = 6

objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*18.1

objpoints = []
imgpoints = []

images = glob.glob('/home/pi/Desktop/mark camera/pictures*.jpg')

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)

    if ret == True:

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 405)
        cv2.imshow('findCorners',img)
        cv2.waitKey(1)

        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)# abc
        print("ret:")
        print(ret)
        print("\nmtx:")
        print(mtx)
        print("\ndist:")
        print(dist)








#print("rvecs:\n",rvecs)
#print("tvecs:\n",tvecs  )