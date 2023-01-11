import cv2
left = cv2.imread('imL.bmp')
right = cv2.imread('imR.bmp')
import numpy as np
from matplotlib import pyplot as plt
window_size = 3
min_disp = 2
num_disp = 162-min_disp
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
grayR= cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
grayL= cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
dispL= disp
dispR= stereoR.compute(grayR,grayL)
filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp

kernel= np.ones((3,3),np.uint8)

closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
dispc= (closing-closing.min())*255
dispC= dispc.astype(np.uint8)
print(dispC[200][200])
depth = (525*8)/dispC[300][200]
print(depth)

                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_HSV) 

    # Show the result for the Depth_image
    #cv2.imshow('Disparity', disp)
    #cv2.imshow('Closing',closing)
    #cv2.imshow('Color Depth',disp_Color)
while True:
 cv2.imshow('Filtered Color Depth',filt_Color)
 cv2.waitKey(1)