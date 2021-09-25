""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


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


def get_gaussian(diff_sq_sum,mu,sigma_sq):
    p = (2*np.pi)**(mu.shape[2]/2)
    exponent = -0.5*diff_sq_sum/sigma_sq
    gaussians = np.exp(exponent)/(p*(sigma_sq**0.5))
    return gaussians

def get_background_gaussians(w,sigma,T):
    w_ratio = -1*w/sigma
    sorted_ratio_idx = np.argsort(w_ratio,axis=2)
    w_ratio.sort(axis=2)
    ratio_cumsum = np.cumsum(-1*w_ratio,axis=2)
    threshold_mask = (ratio_cumsum < T)
    background_gaussian_mask = np.choose(np.rollaxis(sorted_ratio_idx,axis=2),np.rollaxis(threshold_mask,axis=2))
    return np.rollaxis(background_gaussian_mask,axis=0,start=3)

def get_masks(background_gaussians_mask,diff_sq_sum,lambda_sq,sigma_sq):
    update_mask = background_gaussians_mask*(diff_sq_sum/sigma_sq < lambda_sq*sigma_sq)
    foreground_mask = ~np.any(update_mask,axis=2)
    mask_img = np.array(foreground_mask*255,dtype=np.uint8)
    replace_mask = np.repeat(foreground_mask[...,None],background_gaussians_mask.shape[2],axis=2)
    return update_mask, replace_mask, mask_img

def update(gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask,frame):
    replace_mask_extended = np.repeat(replace_mask[:,:,None,:],mu.shape[2],axis=2)
    update_mask_extended = np.repeat(update_mask[:,:,None,:],mu.shape[2],axis=2)
    w = (1-alpha)*w + alpha*update_mask
    w[replace_mask] = 0.0001
    rho = alpha*gaussians
    rho_extended = np.repeat(rho[:,:,None,:],mu.shape[2],axis=2)
    mu[update_mask_extended] = (1-rho_extended[update_mask_extended])*mu[update_mask_extended] + rho_extended[update_mask_extended]*np.repeat(frame[...,None],mu.shape[3],axis=3)[update_mask_extended]
    mu[replace_mask_extended] = np.repeat(frame[...,None],mu.shape[3],axis=3)[replace_mask_extended]
    sigma_sq[replace_mask] = 16
    sigma_sq[update_mask] = (1-rho[update_mask])*sigma_sq[update_mask] + rho[update_mask]*diff_sq_sum[update_mask]
    sigma = np.sqrt(sigma_sq)
    return w, mu, sigma_sq, sigma

def baseline_bgs(args):
    #TODO complete this function
    
    frames = os.listdir(args.inp_path)
    evals = os.listdir('COL780-A1-Data/baseline/groundtruth')
    frames.sort()
    evals.sort()

    bgsub1 = cv2.createBackgroundSubtractorMOG2()
    bgsub2 = cv2.createBackgroundSubtractorKNN()

    # Initial values of parameters
    frame = cv2.imread(args.inp_path+'/'+frames[0])
    K = 3
    lambda_sq = 2.5**2
    alpha = 0.2
    T = 0.7
    w = np.full((frame.shape[0],frame.shape[1],K),1/K)
    mu = np.zeros(frame.shape+tuple([K]))
    sigma = np.ones(w.shape)
    sigma_sq = sigma
    diff = frame[...,None] - mu
    diff_sq_sum = np.sum(diff*diff,axis=2)


    f=open(args.eval_frames,'r')
    data = f.read()
    print(data,type(data))
    data = data.split(' ')
    start_frame = int(data[0])
    end_frame = int(data[1])

    i = 0

    # return
    for a in frames:
        
        
        out_file_name = frame
        

        # img = cv2.imread('shadows.png', -1)


        # cv2.imwrite('shadows_out.png', result)
        # cv2.imwrite('shadows_out_norm.png', result_norm)
        eval = cv2.imread('COL780-A1-Data/baseline/groundtruth/'+evals[i])
        
        gaussians = get_gaussian(diff_sq_sum,mu,sigma_sq)
        background_gaussians_mask = get_background_gaussians(w,sigma,T)
        update_mask, replace_mask, mask_img = get_masks(background_gaussians_mask,diff_sq_sum,lambda_sq,sigma_sq)
        mask1 = bgsub1.apply(frame)
        mask2 = bgsub2.apply(frame)
        cv2.imshow('Input',frame)
        cv2.imshow('Mask',mask_img)
        cv2.imshow('GMM_OpenCV',mask1)
        cv2.imshow('KNN_OpenCV',mask2)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # ret, frame = vid.read()
        frame = cv2.imread(args.inp_path+'/'+a)
        diff = frame[...,None] - mu
        diff_sq_sum = np.sum(diff*diff,axis=2)
        w, mu, sigma_sq, sigma = update(gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask,frame)    
        
            

        i+=1
        
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
        # cv2.imshow('image',frame)
    cv2.destroyAllWindows()
    pass


def illumination_bgs(args):
    #TODO complete this function
    pass


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    #TODO complete this function
    pass


def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
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