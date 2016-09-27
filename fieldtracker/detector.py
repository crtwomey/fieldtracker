
import numpy as np
import cv2
import math
import sys
import csv
from distutils.util import strtobool


# Ensure we have a boolean value from parameter dictionary
def boolify(v):
    if isinstance(v, basestring):
        return strtobool(v) == 1
    return v


class Detector:
    parameter_defaults = {
        "min-threshold"         : 0,
        "max-threshold"         : 256,
        "filter-by-color"       : True,
        "blob-color"            : 255,
        "filter-by-area"        : True,
        "min-area"              : 100,
        "max-area"              : 1500,
        "filter-by-circularity" : True,
        "min-circularity"       : 0.0001,
        "filter-by-convexity"   : True,
        "min-convexity"         : 0.00001,
        "filter-by-inertia"     : True,
        "min-inertia-ratio"     : 0.0001,
        "max-inertia-ratio"     : 0.9,
        "subtract-background"   : False,
        "bg-file-y"             : 'bg_y.png',
        "bg-file-uv"            : 'bg_uv.png',
        "use-mask"              : False,
        "mask-file"             : 'mask.png'
    }

    def __init__(self, detection_filename, params):
        self.params = params

        # setup detection writer
        self.detection_file   = open(detection_filename, 'w')
        self.detection_writer = csv.writer(self.detection_file, delimiter=',')
    
        # BLOB DETECTION 
        # Setup SimpleBlobDetector parameters.
        det = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        det.minThreshold = float(params['min-threshold'])
        det.maxThreshold = float(params['max-threshold'])
        
        # Specify dark color
        det.filterByColor = boolify(params['filter-by-color'])
        det.blobColor     = int(params['blob-color'])
        
        # Filter by Area.
        det.filterByArea = boolify(params['filter-by-area'])
        det.minArea      = float(params['min-area'])
        det.maxArea      = float(params['max-area'])
        
        # Filter by Circularity
        det.filterByCircularity = boolify(params['filter-by-circularity'])
        det.minCircularity      = float(params['min-circularity'])
        
        # Filter by Convexity
        det.filterByConvexity = boolify(params['filter-by-convexity'])
        det.minConvexity      = float(params['min-convexity'])
        
        # Filter by Inertia
        det.filterByInertia = boolify(params['filter-by-inertia'])
        det.minInertiaRatio = float(params['min-inertia-ratio'])
        det.maxInertiaRatio = float(params['max-inertia-ratio'])

        # Ensure conversion to boolean for later use
        self.params['subtract-background'] = boolify(params['subtract-background'])
        self.params['use-mask']            = boolify(params['use-mask'])
        
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            self.detector = cv2.SimpleBlobDetector(det)
        else: 
            self.detector = cv2.SimpleBlobDetector_create(det)
        
        # choose whether to subtract background images or not.
        if self.params['subtract-background']:
            self.background_image_y  = cv2.imread(
                params['bg-file-y'], cv2.CV_LOAD_IMAGE_GRAYSCALE
            )
            self.background_image_uv = cv2.imread(
                params['bg-file-uv'], cv2.CV_LOAD_IMAGE_GRAYSCALE
            )
        
        # mask file
        self.mask = None
        if self.params['use-mask']:
            self.mask = cv2.imread(params['mask-file'], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        
        # linear combination of u and v image components
        self.uvmix = np.array([[0.15, 0.85]]).T
    
    
    def close(self):
        # finish writing detections
        if self.detection_file != None:
            self.detection_file.close()
            self.detection_file = None

    
    def detect(self, time, frame):
        # convert to YUV colorspace
        img1_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # split channels
        yraw, uraw, vraw = cv2.split(img1_cvt)
        # linear mixture of u and inverted v channels
        uvraw = np.array(uraw * self.uvmix[0] + (255-vraw) * self.uvmix[1], dtype='uint8')
        if self.params['subtract-background']:
            yraw  = yraw  - self.background_image_y + 128
            uvraw = uvraw - self.background_image_uv + 128
        # Gaussian blur
        gsize  = 21
        yblur  = cv2.GaussianBlur(yraw,  (gsize,gsize), 0)
        uvblur = cv2.GaussianBlur(uvraw, (gsize,gsize), 0)
        # simple thresholding
        thresh = [82, 120, 115, 135]
        _, y   = cv2.threshold(yblur, thresh[0], 255, cv2.THRESH_BINARY_INV)
        _, uv  = cv2.threshold(uvblur, thresh[3], 255, cv2.THRESH_BINARY)
        # combine thresholded frames and apply "or" command to keep either type of fish
        yuv = np.minimum(y + uv, 255)
        # combine with mask to eliminate regions outside roi
        if self.mask != None:
            mask_inv = cv2.bitwise_not(self.mask)
            yuv      = cv2.bitwise_and(yuv, yuv, mask = self.mask)
        # detect blobs
        keypoints = self.detector.detect(yuv)
        # record detection
        for k in keypoints:
            x, y = k.pt
            detection = [time, x, y]
            self.detection_writer.writerow(detection)
        #convert back to bgr
        yuv_bgr = cv2.cvtColor(yuv, cv2.COLOR_GRAY2BGR)
        # return the blob centerpoints and the processed image
        return (keypoints, yuv_bgr)
    

    def draw_keypoints(self, keypoints, frame):
        #now write out the positions of each detection at each timepoint for using Munkers (Hungarian Matching) Algorithm
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        return cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# gui = None        : no display
#     = 'processed' : show detections with image processing applied to frame
#     = 'original'  : show detections with original frame
def run_detections(detector, video_in, video_out=None, gui=None):
    # load video to extract properties and then close it
    print "loading", video_in
    cap = cv2.VideoCapture(video_in)

    # read out dimensions of video
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    print "input video :", width, "x", height, "pixels"

    # get total frame count (not always reliable)
    #totframes = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # manually count number of frames
    totframes = 0
    ret       = True
    while ret:
        ret, frame = cap.read()
        if ret:
            totframes = totframes + 1
    print "            :", totframes, "frames"

    # close and reopen the video to start at the beginning
    cap.release()
    cap = cv2.VideoCapture(video_in)

    # create video output if requested
    out = None
    if video_out != None:
        # mp4v seems to be only codec that will work
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        out    = cv2.VideoWriter()
        # format must be width, height; last argument specifies whether there is color or not
        out.open(video_out,fourcc, 25.0, (width,height), 0)
        if not out:
            print "!!! Failed VideoWriter: invalid parameters"
            syst.exit(1)
    
    # run detections on each frame
    for i in range(0,totframes):
        # get the next frame
        _, frame = cap.read()
         
        print (i + 1), 'of', totframes, 'frames'
        
        keypoints, processed = detector.detect(i, frame)
        
        if out != None:
            img = detector.draw_keypoints(keypoints, frame)
            out.write(img)
       
        if gui == 'processed':
            img = detector.draw_keypoints(keypoints, processed)
            cv2.imshow('detections', img)
        elif gui == 'original':
            img = detector.draw_keypoints(keypoints, frame)
            cv2.imshow('detections', img)
        if (gui != None) and (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    
    # cleanup
    cap.release()
    if out != None:
        out.release()
    if gui != None:
        cv2.destroyAllWindows()
        cv2.waitKey(1)

