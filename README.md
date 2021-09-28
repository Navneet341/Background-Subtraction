# Background-Subtraction

Background subtraction is a well-known technique for extracting the foreground objects in images or videos. The main aim of background subtraction is to separate moving object foreground from the background in a video, which makes the subsequent video processing tasks easier and more efficient. Usually, the foreground object masks are obtained by performing a subtraction between the current frame and a background model. The background model contains the characteristics of the background or static part of a scene. Figure 1 shows a high level schematic of the background subtraction technique. In this Repositery, We will be using background subtraction to find the foreground object masks for different scene conditions such as varying illumination, jitter, moving background and pan tilt zoom.

Analysis and approach used to identify foreground mask is in details in [Report](https://github.com/Navneet341/Background-Subtraction/blob/main/Background%20Subtraction.pdf)

## Pre-requisites:
1) Python3 Compiler(Preferably the latest version) (If not refer https://www.python.org/downloads/)
2) OpenCV
3) Python3



### To install OpenCV refer:

i. windows: https://www.geeksforgeeks.org/how-to-install-opencv-for-python-in-windows/

ii. MacOS: https://learnopencv.com/install-opencv3-on-macos/


## Execution of Code

Clone Repo on your local machine
Install the dependencies and devDependencies given above.

```sh
python main.py  --inp_path=<path to input frames> \
                --out_path=<path to generated masks> \
                --eval_frames=<path to eval_frames.txt file> \
                --category="<b/i/j/m/p>"
```
- inp_path => provide path containing input frames
- out_path => provide path where output foremask frames are to be saved
- eval_frames => contains starting and ending index of evaluation frame
- Category => category argument is used for determining the scene category. b, i, j, m, and p refer to baseline, illumination, jitter, moving background (dynamic scene), and PTZ categories, respec- tively


For evaluation on basis of IOU values...

```sh
python eval.py  --pred_path=<path to generated masks folder> 
                --gt_path=<path to groundtruth masks folder>
```


## Credits
- [ramrap](https://github.com/ramrap) 
- [Navneet341](https://github.com/Navneet341)





