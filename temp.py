""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def read_eval_data(path):
    f=open(path,'r')
    data = f.read()
    print(data,type(data))
    data = data.split(' ')
    start_frame = int(data[0])
    end_frame = int(data[1])

    return start_frame



def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def baseline_bgs(args):
    #TODO complete this function
    
    frames = os.listdir(args.inp_path)
    evals = os.listdir('COL780-A1-Data/baseline/groundtruth')
    frames.sort()
    evals.sort()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bgsub2 = cv2.createBackgroundSubtractorKNN()

   

    f=open(args.eval_frames,'r')
    data = f.read()
    print(data,type(data))
    data = data.split(' ')
    start_frame = int(data[0])
    end_frame = int(data[1])

    i = 0

    # return
    for frame in frames:
        
        
        out_file_name = frame
        frame = cv2.imread(args.inp_path+'/'+frame)
        
        eval = cv2.imread('COL780-A1-Data/baseline/groundtruth/'+evals[i])
        
        mask2 = bgsub2.apply(frame)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.medianBlur(mask2, 9)
        ret, mask2 = cv2.threshold(mask2, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Input',frame)
        cv2.imshow('KNN_OpenCV',mask2)
        cv2.imshow("Eval",eval)

        if(i>=(start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+=1 
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    cv2.destroyAllWindows()
    pass


def illumination_bgs(args):
    #TODO complete this function

    frames = os.listdir(args.inp_path)
    evals = os.listdir('COL780-A1-Data/illumination/groundtruth')
    frames.sort()
    evals.sort()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    bgsub2 = cv2.createBackgroundSubtractorMOG2(history= 30, varThreshold= 30, detectShadows=True)
    bgsub1 = cv2.createBackgroundSubtractorKNN()

    start_frame = read_eval_data(args.eval_frames)

    i = 0
    for frame in frames:
        out_file_name = frame

        frame = cv2.imread(args.inp_path+'/'+frame)
        eval = cv2.imread('COL780-A1-Data/illumination/groundtruth/'+evals[i])

        width = int(eval.shape[1])
        height = int(eval.shape[0])
        dim = (width, height)
        frame= cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frame = clahe.apply(frame)

        mask2 = bgsub2.apply(frame)
        kernel1 = np.ones((3,3), np.uint8)
        ret, mask2 = cv2.threshold(mask2, 140, 255, cv2.THRESH_BINARY)
        mask2 = cv2.dilate(mask2, kernel1, iterations=1)
        mask2 = cv2.medianBlur(mask2, 3)
        mask2 = cv2.medianBlur(mask2, 3)
        mask2 = cv2.medianBlur(mask2, 3)
       

        print('dim',dim)
        cv2.imshow('Input Illum',frame)
        cv2.imshow('KNN_OpenCV',mask2)
  

        if(i>=(start_frame)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+=1 
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    cv2.destroyAllWindows()

    pass


def jitter_bgs(args):
    #TODO complete this function


    
    pass


def dynamic_bgs(args):
    #TODO complete this function

    frames = os.listdir(args.inp_path)
    evals = os.listdir('COL780-A1-Data/moving_bg/groundtruth')
    frames.sort()
    evals.sort()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bgsub2 = cv2.createBackgroundSubtractorKNN()
    
    f=open(args.eval_frames,'r')
    data = f.read()
    print(data,type(data))
    data = data.split(' ')
    start_frame = int(data[0])
    end_frame = int(data[1])

    i = 0

    # return
    for frame in frames:
        
        
        out_file_name = frame
        frame = cv2.imread(args.inp_path+'/'+frame)
        
        eval = cv2.imread('COL780-A1-Data/moving_bg/groundtruth/'+evals[i])
        
        mask2 = bgsub2.apply(frame)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.medianBlur(mask2, 9)
        ret, mask2 = cv2.threshold(mask2, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow('Input',frame)
        cv2.imshow('KNN_OpenCV',mask2)
        cv2.imshow("Eval",eval)

        if(i>=(start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+=1 
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    cv2.destroyAllWindows()
    pass
    


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    print(args)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    print('Run')
    args = parse_args()
    print('args')
    main(args)