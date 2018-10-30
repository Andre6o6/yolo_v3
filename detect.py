from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
                        default = "weights/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = 416, type = int)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

def load_model(args, CUDA):
    #load model config and weights 
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    
    #set resolution
    model.net_info["height"] = args.reso
    assert args.reso % 32 == 0 
    assert args.reso > 32
    
    #if there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    #set the model in evaluation mode
    model.eval()
    
    return model

def draw_bbox(x, results):
    '''
    Draws bounding box rectangle and rectangle with class name
    '''
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def view_image(image):
    #TODO
    pass

# Parse arguments
args = arg_parse()

images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)

verbose = False #FIXME to arguments
FORMATS = {'.jpg', '.jpeg', '.png'}

CUDA = torch.cuda.is_available()
if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

num_classes = 80    #For COCO
classes = load_classes("data/coco.names")

colors = pkl.load(open("pallete", "rb"))

#Set up the neural network
print("Loading network.....")
model = load_model(args, CUDA)
print("Network successfully loaded")

inp_dim = int(model.net_info["height"])

start_time = time.time()    #checkpoint used to measure time



# Detection phase
read_dir_time = time.time()

# Load image directories
imlist = []
if os.path.isdir(images):
    imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
elif os.path.isfile(images):
    ext = os.path.splitext(images)[1]
    if ext not in FORMATS:
        print("Not a supported image")
        exit()
    imlist.append(os.path.join(os.path.realpath('.'), images))
else:
    print ("No file or directory with the name {}".format(images))
    exit()
    
# Create dir for processed images
if not os.path.exists(args.det):
    os.makedirs(args.det)
    
read_dir_time = time.time() - read_dir_time



# Load images
load_batch_time = time.time()    

loaded_ims = [cv2.imread(x) for x in imlist]    # Numpy arrays for images in BGR

#PyTorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

#List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
    im_dim_list = im_dim_list.cuda()
    
#Create the batches
leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                       len(im_batches))]))  for i in range(num_batches)]
    
load_batch_time = time.time() - load_batch_time


#Loop through batches
output = torch.empty(0)
no_detections = 0
det_loop_time = time.time()

for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad():   #autograd will not compute gradients in forward pass
        prediction = model(Variable(batch), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thresh)

    end = time.time()

    if len(prediction)==0:    #Objects not detected
        no_detections += 1
        continue

    prediction[:,0] += i*batch_size    #transform the attribute from index in batch to index in imlist 

    output = torch.cat((output,prediction))     #concatinate prediction to result tensor
    
    if verbose:
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()  #make sure CUDA returns control to CPU only after GPU work is done

        
det_loop_time = time.time() - det_loop_time

if len(output)==0:
    print ("No detections were made")
    exit()

if no_detections != 0:
    print("No detections on {} images.".format(no_detections))


output_recast_time = time.time()

#Transform the co-ordinates of the boxes to be measured with respect to boundaries of the area 
#on the padded image that contains the original image
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

#Undo the rescaling
output[:,1:5] /= scaling_factor

#Clip any bboxes that have boundaries outside the image
#TODO vectorize mb?
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
output_recast_time = time.time() - output_recast_time


# Draw bounding boxes
draw_time = time.time()

list(map(lambda x: draw_bbox(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

list(map(cv2.imwrite, det_names, loaded_ims))
draw_time = time.time() - draw_time

end_time = time.time()


# Print summary
print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses",read_dir_time))
print("{:25s}: {:2.3f}".format("Loading batch", load_batch_time))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", det_loop_time))
print("{:25s}: {:2.3f}".format("Output Processing", output_recast_time))
print("{:25s}: {:2.3f}".format("Drawing Boxes", draw_time))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end_time - start_time)/len(imlist)))
print("----------------------------------------------------------")
print("No detections on {} images".format(no_detections))

torch.cuda.empty_cache()