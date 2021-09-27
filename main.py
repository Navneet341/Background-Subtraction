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

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)





def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1


    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

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

    # intial = []
    # for frame in frames[:start_frame]:
    #     frame = cv2.imread(args.inp_path+'/'+frame)
    #     intial.append(frame)


    # medianFrame = np.median(intial, axis=0).astype(dtype=np.uint8)   
   


    # return
    for frame in frames:
        print(i)
        
        
        out_file_name = frame
        frame = cv2.imread(args.inp_path+'/'+frame)
        eval = cv2.imread('COL780-A1-Data/baseline/groundtruth/'+evals[i])

        mask2 = bgsub2.apply(frame)
        contours , temp = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((mask2.shape[0:2]),dtype= np.uint8)
        mask = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

        for cnt in range(len(contours)):
            if len(contours[cnt]) > 5:
                tt = []
                for c in contours[cnt]:
                    tt.append(c[0])
                cv2.fillPoly(mask,[np.array(tt)], (255,255,255))


        mask2 = mask
       
        mask2= cv2.erode(mask2, kernel, iterations=1)
        mask2 = cv2.medianBlur(mask2, 15)
        ret, mask2 = cv2.threshold(mask2, 100, 255, cv2.THRESH_BINARY)
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        cv2.imshow('KNN_OpenCV',mask2)
        # mask2 = cv2.filter2D(mask2, -1, kernel)
        cv2.imshow('Input',frame)
        # cv2.imshow('med',medianFrame)
        cv2.imshow('sharped',mask2)
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
    bgsub1 = cv2.createBackgroundSubtractorKNN()
    
    f=open(args.eval_frames,'r')
    data = f.read()
    print(data,type(data))
    data = data.split(' ')
    start_frame = int(data[0])
    end_frame = int(data[1])

    i = 0

    for frame in frames:
        
        
        out_file_name = frame
        frame = cv2.imread(args.inp_path+'/'+frame)
        gausframe = cv2.GaussianBlur(frame, (5,5),0)

        # frame, alpha, beta = automatic_brightness_and_contrast(frame)
        eval = cv2.imread('COL780-A1-Data/moving_bg/groundtruth/'+evals[i])
    
        mask2 = frame
        mask2 = bgsub2.apply(frame)
        mask1 = bgsub1.apply(gausframe)
        cv2.imshow('after bgsub',mask2)
        cv2.imshow('gaus bgsub',mask1)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        # mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('after morp',mask2)
        mask2 = cv2.medianBlur(mask2, 15)
        cv2.imshow('after median blur',mask2)
        ret, mask2 = cv2.threshold(mask2, 130, 255, cv2.THRESH_BINARY)
       
        cv2.imshow('Input',frame)
        cv2.imshow('Gaus frame',gausframe)
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