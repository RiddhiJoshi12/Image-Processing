from __future__ import print_function
import numpy as np
import os
import cv2 as cv
import time
from PIL import ImageFilter
from PIL import Image
from PIL.ExifTags import TAGS
from matplotlib import pyplot as plt
from datetime import datetime
from dateutil import tz
import csv
#from skimage import feature

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 3:
                return True
            elif i==row1-1 and j==row2-1:
                return False

#*************************************Code of Renaming and deriving time info*****************************************

src_path ='E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\19\\'
dst_path ='E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\FG_19\\'
cropped_path = 'E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\Cropped_19\\'
masked_path = 'E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\Masked_19\\'
#contour_path = 'E:\\Acadia\\Traffic Density\\Data\\Weather Code\\Webcam Data\\argylestreet_20181219\\2018\\12\\Contours 3.0\\'
rename_path = 'E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\Renamed_19\\'
path_of_day = 'E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\'

time_info_file = 'timeinfo_file_20181219.csv'
prediction_data_dile = 'prediction_data_20181219.csv'

if not os.path.exists(dst_path):
    os.makedirs(dst_path)
if not os.path.exists(cropped_path):
    os.makedirs(cropped_path)
if not os.path.exists(masked_path):
    os.makedirs(masked_path)
#if not os.path.exists(contour_path):
#    os.makedirs(contour_path)
if not os.path.exists(rename_path):
    os.makedirs(rename_path)


def Rename_and_Time():

    number = 0
    prev_time = 000000

    with open(path_of_day + time_info_file, mode='w') as time_file:
        time_writer = csv.writer(time_file, delimiter=',', lineterminator='\n')

        os.chdir(src_path)
        for file in os.listdir():
            new_path = os.path.join(src_path , file)
            os.chdir(new_path)
            for file in os.listdir():
                
                f = open(file,'rb+')
                file_contents = f.read()
                file_contents = str(file_contents)
                time_of_frame = file_contents[file_contents.find('nTIM=')+5:file_contents.find('\\r\\nTZN=')]
                time = file_contents[file_contents.find('TIT=')+4:file_contents.find('\\r\\nENO=')]

                current_time = int(time[:time.find('.')])*1000 + int(time[time.find('.')+1:])
                diff = current_time - prev_time
                prev_time = current_time

                frame = 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'
                #To remove date part and convert it into H:M:S.mm format
                time_of_frame = datetime.strptime(time_of_frame, '%H:%M:%S.%f')
                #To print time in H:M:S format in csv file 
                time = datetime.strftime(time_of_frame, '%H:%M:%S')
                hour = datetime.strftime(time_of_frame, '%H')


                time_writer.writerow([frame, time, hour, diff])
                
                print(frame, time_of_frame, diff)
                
                frame = cv.imread(file)
                if frame is not None:
                    cv.imwrite(os.path.join(rename_path, 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), frame)
            
                number+=1
    time_file.close()
    #close('E:\\Acadia\\Traffic Density\\Data\\Weather Code\\Webcam Data\\argylestreet_20181219\\2018\\12\\timeinfo_file_20181219.csv')

    return True

def prediction_data_prep():

    if os.path.exists(path_of_day + time_info_file):
        with open(path_of_day + time_info_file, newline='') as csvfile:
           timeinfofile = list(csv.reader(csvfile))
    else:
        timeinfofile = [[]]
    
    time_diff_data = np.array(timeinfofile)

    _file = open(path_of_day + prediction_data_dile, mode='w')
    _writer = csv.writer(_file, delimiter=',', lineterminator='\n')
    row_count = sum(1 for row in time_diff_data)

    for i_row in range(row_count):
        
        if i_row == 0:
            continue
        
        else:
            frame_1 = time_diff_data[i_row-1][0]
            frame_2 = time_diff_data[i_row][0]
                               #Previous Frame           Current Frame          Label   Previous Time              Current Time
            _writer.writerow([frame_1[:22] + "fg-" + frame_1[-9:], frame_2[:22] + "fg-" + frame_2[-9:], 1, time_diff_data[i_row-1][3], time_diff_data[i_row][3]])
    
    _file.close()
    #close('E:\\Acadia\Traffic Density\\Data\\Weather Code\\CNN_RNN\\Train180\\New folder\\' + csv_file)

    return True

#*********************************Code of Masking, Opening, Multiplication*****************************

# os.chdir(src_path)

# for file in os.listdir():
#     new_path = os.path.join(src_path , file)
#     os.chdir(new_path)
#     for file in os.listdir():
#         img = cv.imread(file)
#         if img is not None:
#             mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

#             #points = np.array([[[82,518],[82,542],[358,450],[358,432]]],dtype=np.int32)
#             points = np.array([[[82,505],[82,542],[358,450],[358,420]]],dtype=np.int32)  #Cropped 3.0
#             cv.fillPoly(mask, points, 255)

#             res = cv.bitwise_and(img,img,mask = mask)
#             rect = cv.boundingRect(points) # returns (x,y,w,h) of the rect
#             cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

#             #cv.imshow("cropped" , cropped )
#             #cv.waitKey(0)
#             # crp_img = img[400:550, 100:550]
#             # cv.imshow('',crp_img)
#             #path = 'E:\\Acadia\\AI\\LML - Thesis\\Weather Code\\Webcam Data\\argylestreet_20181219\\2018\\12\\12\\Cropped\\'
#             cv.imwrite(os.path.join(dst_path , 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), cropped)
#             number+=1
##---------------------------------------------------------------------------------------------------------------------------------------------------
def masking_of_images():

    number = 0
    capture = cv.VideoCapture('E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\Renamed_19\\argylestreet_20181219-%05d.jpg')

    if not capture.isOpened():
        print('Unable to open Images')
        exit(0)

    backSub = cv.bgsegm.createBackgroundSubtractorMOG(history=2, nmixtures = 5, backgroundRatio = 0.007, noiseSigma = 10) #++
    backSub1 = cv.createBackgroundSubtractorMOG2(history=2, varThreshold=100, detectShadows=0)
    backSub1 = cv.bgsegm.createBackgroundSubtractorMOG(history=2, nmixtures = 5, backgroundRatio = 0.20, noiseSigma = 50)
    backSub2 = cv.bgsegm.createBackgroundSubtractorMOG(history=2, nmixtures = 5, backgroundRatio = 0.005, noiseSigma = 0)

    while True:
    
        ret, frame = capture.read()
    
        if frame is None:
            break

        #-------------------------------Code to Crop while Masking--------------------------
        #img=frame
        
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        #points = np.array([[[594,400],[594,595],[745,595],[745,400]]],dtype=np.int32) (Old co-ordinate)
        points = np.array([[[642,376],[692,381],[679,486],[621,478]]],dtype=np.int32)
            
        cv.fillPoly(mask, points, 255)
        res = cv.bitwise_and(frame,frame,mask = mask)
        rect = cv.boundingRect(points) # returns (x,y,w,h) of the rect
        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        cv.imwrite(os.path.join(cropped_path , 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), cropped)
     

        ###############-w
        #fgMask = backSub.apply(frame, 1)
        #fgMask1 = backSub1.apply(frame, 0.005)
        fgMask2 = backSub2.apply(frame, 0.005)

        # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        ###########################################################

        ################-Opening
        kernel = np.ones((6,6))
        kernel1 = np.ones((3,3))
        opening = cv.morphologyEx(fgMask2, cv.MORPH_OPEN, kernel1)
        #kernel = np.ones((3,3))
        #opening = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(fgMask2, cv.MORPH_CLOSE, kernel)

        ###########################################################
        cv.imshow('Frame', fgMask2)
        #cv.imshow('MOG', opening)
        #cv.imshow('MOG 2', closing)
        #cv.imshow('MOG 3', fgMask2)
       # cv.imwrite(os.path.join('E:\\Graduate_CS\\Thesis_Work\\Data\\argylestreet_20181219\\2018\\12\\Processed_19\\' , 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), fgMask2)
       # number+=1
        #img=fgMask2
        #mask = np.zeros((fgMask2.shape[0], fgMask2.shape[1]), dtype=np.uint8)
        #points = np.array([[[82,505],[82,542],[358,450],[358,420]]],dtype=np.int32)
        #cv.fillPoly(mask, points, 255)
        #res = cv.bitwise_and(fgMask2, fgMask2, mask = mask)
        #rect = cv.boundingRect(points) # returns (x,y,w,h) of the rect
        #masked = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

        cv.imwrite(os.path.join(masked_path, 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), fgMask2)
        number += 1
    return True


def multiply_image():
    number = 0
    os.chdir(masked_path)

    for file in os.listdir():
        new_path = os.path.join(masked_path , file)
        file = file.replace("-cnt","")
        Contour = cv.imread(new_path)
        Cropped = cv.imread(os.path.join(cropped_path, file))
    
        Contour = Contour.astype(float)
        Cropped = Cropped.astype(float)

        Contour = Contour.astype(float)/255

        falcon = cv.multiply(Contour, Cropped)

        cv.imwrite(os.path.join(dst_path , 'argylestreet_20181219-'+str(number).zfill(5)+'.jpg'), falcon)

        number+=1

    return True

if Rename_and_Time():
   print("Renamed Successfully.")

if prediction_data_prep():
    print("Prediction data prep Successfully.")

if masking_of_images():
    print("Cropped Successfully.")

if multiply_image():
    print("Multiplied Successfully.")
