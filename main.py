""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def read_eval_data(path):
    f= open(path,'r')
    data=  f.read()
    print(data,type(data))
    data=  data.split(' ')
    start_frame=  int(data[0])
    end_frame=  int(data[1])
    return start_frame


def parse_args():
    parser=  argparse.ArgumentParser(description= 'Get mIOU of video sequences')
    parser.add_argument('-i','--inp_path',type= str,default= 'input',required= True,\
                                                        help= "Path for the input images folder")
    parser.add_argument('-o','--out_path',type= str,default= 'result',required= True,\
                                                        help= "Path for the predicted masks folder")
    parser.add_argument('-c','--category',type= str,default= 'b',required= True,\
                                                        help= "Scene category. One of baseline,illumination,jitter,dynamic scenes,ptz (b/i/j/m/p)")
    parser.add_argument('-e','--eval_frames',type= str,default= 'eval_frames.txt',required= True,\
                                                        help= "Path to the eval_frames.txt file")
    args=  parser.parse_args()
    return args


def baseline_bgs(args):
    #TODO complete this function
    
    frames=  os.listdir(args.inp_path)
    evals=  os.listdir('COL780-A1-Data/baseline/groundtruth')
    frames.sort()
    evals.sort()

    kernel=  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bgsub2=  cv2.createBackgroundSubtractorKNN(detectShadows=  False)

    f= open(args.eval_frames,'r')
    data=  f.read()
    print(data,type(data))
    data=  data.split(' ')
    start_frame=  int(data[0])
    end_frame=  int(data[1])

    i=  0
    for frame in frames:
        print(i)
                
        out_file_name=  frame
        frame=  cv2.imread(args.inp_path+'/'+frame)
        eval=  cv2.imread('COL780-A1-Data/baseline/groundtruth/'+evals[i])

        mask2=  bgsub2.apply(frame)
        mask2=  cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernel)
        contours ,temp=  cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        
        # cv2.imshow('bef contour',mask2)
        
        mask=  np.zeros((mask2.shape[0:2]),dtype=  np.uint8)
        mask=  cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR)

        for cnt in range(len(contours)):
            if len(contours[cnt]) > 5:
                tt=  []
                for c in contours[cnt]:
                    tt.append(c[0])
                cv2.fillPoly(mask,[np.array(tt)],(255,255,255))
            else:
                tt=  []
                for c in contours[cnt]:
                    tt.append(c[0])
                cv2.fillPoly(mask,[np.array(tt)],(0,0,0))

        mask2=  mask
      

        kernel1=  np.ones((3,3),np.uint8)
        mask2=  cv2.erode(mask2,kernel1,iterations= 1)
        mask2=  cv2.medianBlur(mask2,15)

        ret,mask2=  cv2.threshold(mask2,100,255,cv2.THRESH_BINARY)
       
        cv2.imshow('Baseline',mask2)
        cv2.imshow('Input',frame)
        if(i>= (start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+= 1 
        keyboard=  cv2.waitKey(30)
        if keyboard==  'q' or keyboard==  27:
            break
    cv2.destroyAllWindows()
    pass


def illumination_bgs(args):
    #TODO complete this function

    frames=  os.listdir(args.inp_path)
    evals=  os.listdir('COL780-A1-Data/illumination/groundtruth')
    frames.sort()
    evals.sort()

    bgsub2=  cv2.createBackgroundSubtractorMOG2(history=  30,varThreshold=  30,detectShadows= True)
    

    start_frame=  read_eval_data(args.eval_frames)
    

    i=  0
    for frame in frames:
        out_file_name=  frame

        frame=  cv2.imread(args.inp_path+'/'+frame)
        eval=  cv2.imread('COL780-A1-Data/illumination/groundtruth/'+evals[i])

        # //////////////
        width=  int(eval.shape[1])
        height=  int(eval.shape[0])
        dim=  (width,height)
        frame=  cv2.resize(frame,dim,interpolation=  cv2.INTER_AREA)
         # //////////////
        

        frame=  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        clahe=  cv2.createCLAHE(clipLimit= 2.0,tileGridSize= (8,8))
        frame=  clahe.apply(frame)

        mask2=  bgsub2.apply(frame)
        kernel1=  np.ones((3,3),np.uint8)
        ret,mask2=  cv2.threshold(mask2,140,255,cv2.THRESH_BINARY)
        mask2=  cv2.dilate(mask2,kernel1,iterations= 1)
        mask2=  cv2.medianBlur(mask2,3)
        mask2=  cv2.medianBlur(mask2,3)
        mask2=  cv2.medianBlur(mask2,3)
       
        cv2.imshow('Input Illumination',frame)
        cv2.imshow('Illumination',mask2)
  
        if(i>= (start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+= 1 
        keyboard=  cv2.waitKey(30)
        if keyboard==  'q' or keyboard==  27:
            break
    cv2.destroyAllWindows()

    pass


def is_contour_bad(c):
	peri=  cv2.arcLength(c,True)
	approx=  cv2.approxPolyDP(c,0.02 * peri,True)
	return  len(approx) < 6


def jitter_bgs(args):
    
    frames=  os.listdir(args.inp_path)
    evals=  os.listdir('COL780-A1-Data/jitter/groundtruth')
    frames.sort()
    evals.sort()

    kernel=  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bgsub2=  cv2.createBackgroundSubtractorKNN(detectShadows=  True)

    start_frame=  read_eval_data(args.eval_frames)
    

    i=  0
    intial=  []
    for frame in frames[:start_frame]:
        frame=  cv2.imread(args.inp_path+'/'+frame)
        intial.append(frame)

    medianFrame=  np.median(intial,axis= 0).astype(dtype= np.uint8)
    # kernel=  np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # medianFrame=  cv2.filter2D(medianFrame,-1,kernel)
    median_img=  medianFrame
    median_img_gray=  cv2.cvtColor(median_img,cv2.COLOR_BGR2GRAY)

    for a in frames:

        out_file_name=  a
        print(a)
        frame=  cv2.imread(args.inp_path+'/'+a)
        # frame=  cv2.filter2D(frame,-1,kernel)
        im1_gray=  cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        

        sz=  median_img.shape
        warp_mode=  cv2.MOTION_TRANSLATION
        if warp_mode==  cv2.MOTION_HOMOGRAPHY :
            warp_matrix=  np.eye(3,3,dtype= np.float32)
        else :
            warp_matrix=  np.eye(2,3,dtype= np.float32)
        number_of_iterations=  5000
        termination_eps=  1e-10
        criteria=  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,number_of_iterations, termination_eps)
        (cc,warp_matrix)=  cv2.findTransformECC (median_img_gray,im1_gray,warp_matrix,warp_mode,criteria)
        if warp_mode==  cv2.MOTION_HOMOGRAPHY :
            im2_aligned=  cv2.warpPerspective (frame,warp_matrix,(sz[1],sz[0]),flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            im2_aligned=  cv2.warpAffine(frame,warp_matrix,(sz[1],sz[0]),flags= cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
        
        # frame=  cv2.GaussianBlur(frame,(5,5),0)
        # frame,alpha,beta=  automatic_brightness_and_contrast(frame)

        eval=  cv2.imread('COL780-A1-Data/jitter/groundtruth/'+evals[i])
        
        mask2=  bgsub2.apply(im2_aligned)
        mask2=  cv2.morphologyEx(mask2,cv2.MORPH_OPEN,kernel)
        # kernel1=  np.ones((3,3),np.uint8)
        # mask2=  cv2.erode(mask2,kernel1,iterations= 1)

        mask2=  cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernel)
        mask2=  cv2.medianBlur(mask2,15)

        ret,mask2=  cv2.threshold(mask2,100,255,cv2.THRESH_BINARY)

        cv2.imshow('Input',frame)
        cv2.imshow('Jitter',mask2)
        # cv2.imshow("Eval",eval)
        if(i>= (start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)

        k=  cv2.waitKey(30) & 0xff
        if k==  27:
            break
        i+= 1
        
        keyboard=  cv2.waitKey(30)
        if keyboard==  'q' or keyboard==  27:
            break
    cv2.destroyAllWindows()
    pass

    # return




def dynamic_bgs(args):
    #TODO complete this function

    frames=  os.listdir(args.inp_path)
    evals=  os.listdir('COL780-A1-data/moving_bg/groundtruth/')
    frames.sort()
    evals.sort()

    
    
    bgsub1=  cv2.createBackgroundSubtractorKNN(detectShadows=  False)
    kernel=  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    start_frame=  read_eval_data(args.eval_frames)
    # print(start_frame)

    i=  0

    for frame in frames:
        
        
        out_file_name=  frame
        frame=  cv2.imread(args.inp_path+'/'+frame)

        gausframe=  cv2.GaussianBlur(frame,(5,5),0)
        eval=  cv2.imread('COL780-A1-data/moving_bg/groundtruth/'+evals[i])
        mask2=  bgsub1.apply(gausframe,kernel,learningRate=  -1)

        mask2=  cv2.morphologyEx(mask2,cv2.MORPH_OPEN,kernel)
        mask2=  cv2.medianBlur(mask2,15)
        ret,mask2=  cv2.threshold(mask2,130,255,cv2.THRESH_BINARY)
        
        contours=  cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        cv2.fillPoly(mask2,contours,255)
        kernel1=  np.ones((7,7),np.uint8)
        mask2=  cv2.erode(mask2,kernel1,iterations= 1)
       
        cv2.imshow('Moving BG',mask2)
        cv2.imshow("Eval",eval)

        if(i>= (start_frame-1)):
            cv2.imwrite(args.out_path+evals[i],mask2)
       
        i+= 1 
        keyboard=  cv2.waitKey(30)
        if keyboard==  'q' or keyboard==  27:
            break
    cv2.destroyAllWindows()
    pass


MAX_FEATURES=  500
GOOD_MATCH_PERCENT=  0.15

def alignImages(im1,im2):


  image_intial=  cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
  image_previous=  cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

  sift_orb=  cv2.ORB_create(MAX_FEATURES)
  keypoints1,descriptors1=  sift_orb.detectAndCompute(image_intial,None)
  keypoints2,descriptors2=  sift_orb.detectAndCompute(image_previous,None)

  
  matcher=  cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches=  matcher.match(descriptors1,descriptors2,None)

  matches.sort(key= lambda x: x.distance,reverse= False)


  numGoodMatches=  int(len(matches) * GOOD_MATCH_PERCENT)
  matches=  matches[:numGoodMatches]


  imMatches=  cv2.drawMatches(im1,keypoints1,im2,keypoints2,matches,None)

  points1=  np.zeros((len(matches),2),dtype= np.float32)
  points2=  np.zeros((len(matches),2),dtype= np.float32)

  for i,match in enumerate(matches):
    points1[i,:]=  keypoints1[match.queryIdx].pt
    points2[i,:]=  keypoints2[match.trainIdx].pt


  h,mask=  cv2.findHomography(points1,points2,cv2.RANSAC)

  
  height,width,channels=  im2.shape
  im1Reg=  cv2.warpPerspective(im1,h,(width,height))

  return im1Reg,h
    


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    frames=  os.listdir(args.inp_path)
    evals=  os.listdir('COL780-A1-Data/ptz/groundtruth')
    frames.sort()
    evals.sort()

    bgsub1=  cv2.createBackgroundSubtractorKNN(history=  5,detectShadows=  False)
    bgsub2=  cv2.createBackgroundSubtractorKNN(history=  5,detectShadows=  False)
    kernel=  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    f= open(args.eval_frames,'r')
    data=  f.read()
    print(data,type(data))
    data=  data.split(' ')
    start_frame=  int(data[0])
    end_frame=  int(data[1])

    i=  0
    prev_10=  cv2.imread(args.inp_path+'/'+frames[0])
    prev=  prev_10

    for frame in frames:
        print(i)


        frame_=  cv2.imread(args.inp_path+'/'+frame)
    

        cv2.imshow('original frame',frame_)

        frame_prev,_=  alignImages(frame_,prev)

        # cv2.imshow('prev alignment',frame_prev)
        
        mask2=  bgsub1.apply(frame_prev,kernel,learningRate=  -1)
        cv2.imshow('mask',mask2)

        
        frame_10,_=  alignImages(frame_,prev_10)
        cv2.imshow('10 alignement',frame_10)
        mask1=  bgsub2.apply(frame_10,kernel,learningRate= -1)
        cv2.imshow('mask1',mask1)

        if(i%10>5):
            mask2=  mask1

    
        mask2=  cv2.morphologyEx(mask2,cv2.MORPH_OPEN,kernel)
        # mask2=  cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernel)
        cv2.imshow('after morph',mask2)
       
        # cv2.imshow('after contour filling',mask2)
        mask2=  cv2.medianBlur(mask2,15)
        contours=  cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        cv2.fillPoly(mask2,contours,255)
        cv2.imshow('after median',mask2)


        ret,mask2=  cv2.threshold(mask2,100,255,cv2.THRESH_BINARY)

        cv2.imshow('final',mask2)

        if(i>= (start_frame-1)):
            
            cv2.imwrite(args.out_path+evals[i],mask2)


        if(i%10==  0):
            prev_10=  frame_

        if (i%10==  5):
            prev=  frame_


        i +=  1
        keyboard=  cv2.waitKey(30)
        if keyboard==  'q' or keyboard==  27:
            break
    cv2.destroyAllWindows()

    pass


def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    print(args)
    FUNCTION_MAPPER=  {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__==  "__main__":
    print('Run')
    args=  parse_args()
    print('args')
    main(args)