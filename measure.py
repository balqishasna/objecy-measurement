import cv2
from numpy import angle
from object_detector import *
import numpy as np
import time
import csv
from bdb import effective
import math


img = cv2.imread("D:\Kuliah\judul-skripsi\codingan\objecy-measurement-master\input\coba.jpg")
img1 = cv2.imread("input\coba.jpg")

#image crop
# y=400
# x=210
# h=800
# w=260
# crop_image = img[x:w, y:h]
# cv2.imshow("Croped", crop_image)
# #cv2.waitKey(0)

list_rect = []

detector = HomogeneousBgDetector()

contours = detector.detect_objects(img)

header = ['file_name', 'x', 'y', 'width' , 'height', 'angle']

with open('data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w , h), angle = rect

    file_name = "hasil_{}.jpg".format(int(time.time()))
    data = []

    data.extend([file_name, w, h])

    list_rect.append(rect)

    print(list_rect, len(list_rect))
    for item in range(len(list_rect)):
      (x, y), (w , h), angle = list_rect[item]
      X = int(math.ceil(x))
      Y = int(math.ceil(y))
      W = int(math.ceil(w))
      H = int(math.ceil(h))
      cropped_image = img[Y:Y+H, X:X+W]
     # cropped_image = img[X:X+W, Y:Y+H]
      # print([x,y,w,h])
      # print(type(x))
      
      #save image cropped
      cv2.imwrite('D:\Kuliah\judul-skripsi\codingan\objecy-measurement-master\output\contour_{}.png'.format(item), cropped_image)
      # print(item)
    with open('data.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(data)


    # display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
#measurement
    #cv2.putText(img, "Width {} px".format(round(w, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 0.6, (100, 200, 0), 1)
    #cv2.putText(img, "Height {} px".format(round(h, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 0.6, (100, 200, 0), 1) 

def BrightnessContrast(brightness=0):
  
  # getTrackbarPos returns the current
  # position of the specified trackbar.
  brightness = cv2.getTrackbarPos('Brightness',
                  'GEEK')
  
  contrast = cv2.getTrackbarPos('Contrast',
                'GEEK')

  effect = controller(img, brightness,
            contrast)

  # The function imshow displays an image
  # in the specified window

#save image
#   cv2.imshow('Effect', effect)
#   k = cv2.waitKey(0) & 0xFF
#   if k == ord('q'):
#     cv2.imwrite("D:\Kuliah\judul-skripsi\codingan\Objecy-measurement-master\output\hasil_{}.jpg".format(int(time.time())), effect)
    
#automation brightness
s = 1024
img = cv2.resize(img, (s,s), 0, 0, cv2.INTER_AREA)

def controller(input_img, brightness = 0, contrast = 0):
#def controller(img, brightness=255, contrast=127):

  #brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

  #contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:
           shadow = brightness
           max = 255
        else:
           shadow = 0
           max = 255 + brightness
        al_pha = (max - shadow) / 255
        ga_mma = shadow

        buf = cv2.addWeighted(input_img, al_pha, input_img, 0, ga_mma)
    else:
        buf = input_img.copy()
    # The function addWeighted calculates
    # the weighted sum of two arrays
    #cal = cv2.addWeighted(img, al_pha,
    #          img, 0, ga_mma)

  #else:
   # cal = img

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
    #Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    #Gamma = 127 * (1 - Alpha)
font = cv2.FONT_HERSHEY_SIMPLEX
fcolor = (0,0,0)

blist = [0] # list of brightness values
clist = [64] # list of contrast values

out = np.zeros((s*2, s*3, 3), dtype = np.uint8)

for i, b in enumerate(blist):
    c = clist[i]
    print('b, c:  ', b,', ',c)
    row = s*int(i/3)
    col = s*(i%3)
    
    print('row, col:   ', row, ', ', col)
    
    out[row:row+s, col:col+s] = controller(img, b, c)
    msg = 'b %d' % b
    cv2.putText(out,msg,(col,row+s-22), font, .7, fcolor,1,cv2.LINE_AA)
    msg = 'c %d' % c
    cv2.putText(out,msg,(col,row+s-4), font, .7, fcolor,1,cv2.LINE_AA)
    
    cv2.putText(out, 'OpenCV',(260,30), font, 1.0, fcolor,2,cv2.LINE_AA)

#cv2.imwrite('out.png', out)

#click 'q' to save
cv2.imshow('Effect', out)
k = cv2.waitKey(0) & 0xFF
if k == ord('q'):
     cv2.imwrite("D:\Kuliah\judul-skripsi\codingan\objecy-measurement-master\output\hasil_{}.jpg".format(int(time.time())), out)
    
    # The function addWeighted calculates
    # the weighted sum of two arrays
#     cal = cv2.addWeighted(cal, Alpha,
#               cal, 0, Gamma)

#   # putText renders the specified text string in the image.
#   cv2.putText(cal, 'B:{},C:{}'.format(brightness,
#                     contrast), (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#   return cal

# if __name__ == '__main__':
#   # # Making another copy of an image.
#   img_copy = img.copy()

#   # The function namedWindow creates a
#   # window that can be used as a placeholder
#   # for images.
#   cv2.namedWindow('GEEK')

#   # The function imshow displays an
#   # image in the specified window.
#   cv2.imshow('GEEK', img_copy)

#   # createTrackbar(trackbarName,
#   # windowName, value, count, onChange)
#   # Brightness range -255 to 255
#   cv2.createTrackbar('Brightness',
#           'GEEK', 255, 2 * 255,
#           BrightnessContrast)
  
#   # Contrast range -127 to 127
#   cv2.createTrackbar('Contrast', 'GEEK',
#           127, 2 * 127,
#           BrightnessContrast)

  
#   BrightnessContrast(0)

# # The function waitKey waits for
# # a key event infinitely or for delay
# # milliseconds, when it is positive.


cv2.waitKey(0)
