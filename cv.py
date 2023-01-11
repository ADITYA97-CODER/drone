
#!/usr/bin/python3

import cv2
import numpy as np
from picamera2 import Picamera2
# Grab images as numpy arrays and leave everything else to OpenCV.
# These values can change depending on the system

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

window_size = 3
min_disp = 2
num_disp = 16
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 2,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
stere = cv2.StereoBM_create()

stere.setNumDisparities(num_disp)
stere.setBlockSize(5)
stere.setTextureThreshold(3)
stere.setUniquenessRatio(11)
stere.setSpeckleRange(16)
stere.setSpeckleWindowSize(100)
stere.setDisp12MaxDiff(1)
stere.setMinDisparity(min_disp)
stereR=cv2.ximgproc.createRightMatcher(stere)


while True:
    im = picam2.capture_array()
    
    right= im[:,0:320]
    left = im[:,320:640]
    
    grayR= cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    disp= stere.compute(grayL,grayR)
    #.astype(np.float32)/ 16

    dispL= disp
    dispR= stereR.compute(grayL,grayR)
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp

    kernel= np.ones((3,3),np.uint8)

    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)
    #print(dispC[200][200])
    #f = (right.shape[1]*0.5) /np.tan(60*0.5*np.pi/180)
    #depth = (8*f)/disp[240,210]
    depth = ((0.8*320)*8)/dispC[240][160]
    print(depth)

                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_PLASMA)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_PLASMA)

    
    
    cv2.imshow("right",im)
    cv2.imshow("disparity",filt_Color)
    cv2.waitKey(1)