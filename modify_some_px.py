#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:44:09 2023

@author: brownianxgames

Remove a undesired region of a movie.
"""

import numpy as np
import cv2
from tqdm import tqdm
import sys
import getopt

# Define the file path and the croping values
# filepath = '/media/brownianxgames/Aquisitions/acquisitions_fares/2023/20231107/8/Basler_acA1920-155um__22392621__20231107_162509284.mp4'
# x,y,w,h = 1050,900,200,200
# new_px_value = 1

# Input auxiliary function
def get_input(argv, x=1050, y=900, w=200, h=200, new_px_value=1):
    # x = x
    # y = y
    # w = w 
    # h = h
    # new_px_value = new_px_value
    opts, args = getopt.getopt(argv,"f:p:",["filepath=","px="])
    for opt, arg in opts:
        if opt in ("-f", "--filepath"):
            filepath = arg
        elif opt in ("-x"):
            x = int(arg)
        elif opt in ("-y"):
            y = int(arg)
        elif opt in ("-w"):
            w = int(arg)
        elif opt in ("-h"):
            h = int(arg)
        elif opt in ("-p", "--px"):
            new_px_value = int(arg)
        else:
            print('Error: Unknown option {}'.format(opt))
            sys.exit()
    return filepath, x, y, w, h, new_px_value


if __name__ == '__main__':
    
    # Input 
    filepath, x, y, w, h, new_px_value = get_input(sys.argv[1:])
    # try:
    #     filepath, x, y, w, h, new_px_value = get_input(sys.argv[1:])
    # except NameError as err:
    #     raise err
    # except:
    #     print('Error but not name')
    
    # Open the video
    vid = cv2.VideoCapture(filepath)

    # Some characteristics from the original video
    w_frame = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize frame counter
    # cnt = 0

    # Initialize output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_name = 'modified.mp4'
    output_name = filepath[:filepath.find('.mp4')] + '_modified.mp4'
    out = cv2.VideoWriter(output_name, fourcc, fps, (w_frame, h_frame))
    # The output is saved in the directory of the input video

    # Cropping starts
    # while(vid.isOpened()):
    for i in tqdm(range(frames)):
    
        ret, frame = vid.read()

        # cnt += 1 # Counting frames

        # Avoid problems when video finish
        if ret==True:
            # Croping the frame
            # temp_frame = frame[y:y+h, x:x+w]
            temp_frame = np.copy(frame)
            temp_frame[y:y+h, x:x+w, :] = new_px_value
        
            # Percentage
            # xx = cnt *100/frames
            # print(int(xx),'%')

            # Save all the video
            out.write(temp_frame)

            # Just to see the video in real time          
            # cv2.imshow('frame',frame)
            # cv2.imshow('croped',crop_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    vid.release()
    out.release()
    # cv2.destroyAllWindows()