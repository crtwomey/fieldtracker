
# Notes:
# July 6th - fixed issue with thresholding between-frame detection assignments.
#          - added relinking phase to sequence together tracklets.
# Aug 24th - AH: added true frame number to 'detections' and 'tracks' as 4th column in array
#          - AH: match tracks to video using original frame number from detections file
#          - AH: fixed bug in detections loop that was dropping first detection for each frame

import numpy as np
import munkres as mk
import cv2
import math
import sys
import csv
import h5py
from sklearn.metrics.pairwise import euclidean_distances
import colorsys


# Returns Munkres assignment for a given cost matrix
#   m : an mk.Munkres object
def find_assignment(m, cost):
    original_cost = np.copy(cost)
    # matching from previous to current
    # NOTE: cost matrix is modified by munkres compute(...)
    assignment = m.compute(cost)
    a = np.array(zip(*assignment)).T
    c = np.array([original_cost[tuple(zip(*assignment))]]).T
    return np.hstack((a,c))


# from http://stackoverflow.com/questions/10901085/range-values-to-pseudocolor/10902473#10902473
def pseudocolor(val, minval, maxval):
    # convert val in range minval..maxval to the range 0..120 degrees which
    # correspond to the colors red..green in the HSV colorspace
    h = (float(val-minval) / (maxval-minval)) * 120
    # convert hsv color (h,1,1) to its rgb equivalent
    # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
    return r*255, g*255, b*255


class Linker:
    parameter_defaults = {
        "max-cost"            : 1000,
        "max-lag"             : 10,
        "cost-threshold"      : 50,
        "relinking-threshold" : 200
    }

    def __init__(self, params):
        self.params = params
        self.params["max-cost"]            = float(params["max-cost"])
        self.params["max-lag"]             = float(params["max-lag"])
        self.params["cost-threshold"]      = float(params["cost-threshold"])
        self.params["relinking-threshold"] = float(params["relinking-threshold"])

        # create assignment solver
        self.munkres = mk.Munkres()
    

    def load_detections(self, detection_filename):
        # open detection file
        detection_file = open(detection_filename, 'r')
        
        # extract some important properties of detections
        # (e.g., on what frame does first detection take place)
        detection_stats_reader = csv.reader(detection_file, delimiter=',')
        firstdetection = next(detection_stats_reader)
        firstframe, firstx, firsty = np.array(firstdetection, dtype='float32')
        firstframe = int(firstframe)
        
        # extract total number of rows and frames
        detection_file = open(detection_filename, 'r')
        detection_numframes_reader = csv.reader(detection_file, delimiter=',')
        self.numrows   = 0
        self.numframes = 0
        
        # read in detection results for each frame
        for row in detection_numframes_reader:
            frame, x, y    = np.array(row, dtype='float32')
            self.numrows   = self.numrows + 1
            self.numframes = int(max((self.numframes,frame)))
        
        # load full detection data
        detection_file        = open(detection_filename, 'r')
        self.detection_reader = csv.reader(detection_file, delimiter=',')
    
    
    def detection_assignments(self):
        current_frame       = 0
        current_detections  = np.empty(shape=(0,3))
        previous_detections = np.empty(shape=(0,3))

        detections  = []
        assignments = []
        
        # determine lowest-cost matches between each set of detections
        detectioncntr = 0
        for detection in self.detection_reader:
            detectioncntr = detectioncntr + 1
            # read each line of csv
            frame, x, y = np.array(detection, dtype='float32')
            # convert frame to integer
            frame = int(frame)
        
            # is the frame of the current line greater than that of the last line?
            if frame > current_frame:
                # append list of detections with all detections from LAST frame    
                detections.append(current_detections)
                if previous_detections.shape[0] > 0:
                    # make cost matrix
                    dist = euclidean_distances(previous_detections, current_detections)
                    cost = np.ones(
                        shape=(max(dist.shape),max(dist.shape)),
                        dtype='float64'
                    ) * self.params["max-cost"]
                    cost[0:dist.shape[0],0:dist.shape[1]] = dist
                    stack = find_assignment(self.munkres, cost)
                    assignments.append(stack)
                previous_detections = current_detections
                # note: AH added this line to replace line at bottom of inner loop
                current_detections = np.array([x, y, frame], ndmin=2)
                current_frame       = current_frame + 1
            else:
                current_detections = np.append(current_detections, [[x, y, frame]], axis=0)
            # use assignments from last frame
            if detectioncntr == self.numrows:
                # append list of detections with detections from CURRENT frame
                detections.append(current_detections)
                if previous_detections.shape[0] > 0:
                    # make cost matrix
                    dist = euclidean_distances(previous_detections, current_detections)
                    cost = np.ones(
                        shape=(max(dist.shape),max(dist.shape)),
                        dtype='float64'
                    ) * self.params["max-cost"]
                    cost[0:dist.shape[0],0:dist.shape[1]] = dist
                    stack = find_assignment(self.munkres, cost)
                    assignments.append(stack)
        
        #restructure detections into an array that is easy to view
        cntr = 0
        for j in range(len(detections)):
            dtemp = detections[j]
            cntr = cntr + dtemp.shape[0]
        
        print '    number of retained detections:', cntr
        print '    number of original detections', self.numrows
        
        cntr = 0
        for j in range(len(assignments)):
            dtemp = assignments[j]
            cntr = cntr + dtemp.shape[0]
        
        print '    number of assignments:', cntr

        self.detections  = detections
        self.assignments = assignments
    
    
    def build_tracklets(self):
        # compute tracks
        tracks = []
        previous_detections_to_tracks = []
        
        # initialize tracks
        p = self.detections[0]
        for j in range(p.shape[0]):
            tracks.append(np.reshape(np.array(np.concatenate(([0], p[j,:], [0]))),(1,5)))
            previous_detections_to_tracks.append(j)
        
        # build tracklets
        for frame in range(len(self.assignments)):
            assignment = self.assignments[frame]
            p = self.detections[frame]
            c = self.detections[frame+1]
            current_detections_to_tracks = [-1]*(assignment.shape[0])
        
            for j in range(assignment.shape[0]):
                # previous is previous assignment
                # current is current assignment
                # cost is the cost associated with the assignment
                previous, current, cost = assignment[j,:]
                # convert previous and current to integers
                previous, current = map(int, [previous, current])
                # WARNING!!! AH 8-31: MAY WANT TO REVISE CONDITION FOR NEW TRACK TO COUNT ALL PREVIOUS TRACK ID NUMBERS
                #if assignment id # in previous is greater than or equal to the number of detections in previous frame, make a new track
                if previous >= p.shape[0]:
                    tracks.append(np.reshape(
                        np.array(np.concatenate([[frame], c[current,:], [0]])), (1,5)
                    ))
                    current_detections_to_tracks[current] = len(tracks)-1
                elif current < c.shape[0]:
                    # start a new track
                    if cost > self.params["cost-threshold"]:
                    	if current >= c.shape[0]:
                            print 'WARNING: current detections contains fewer than previous'
                    	else:
                            tracks.append(np.reshape(
                                np.array(np.concatenate([[frame], c[current,:], [0]])),(1,5)
                            ))
                            current_detections_to_tracks[current] = len(tracks)-1
                            # append to existing track
                    else:
                        tid = previous_detections_to_tracks[previous]
                        tracks[tid] = np.vstack((
                            tracks[tid], np.concatenate([[frame], c[current,:], [cost]])
                        ))
                        current_detections_to_tracks[current] = tid
                        # if current >= c.shape[0], then some previous track is not getting a new assignemnt,
                        # so no need to do anything
            previous_detections_to_tracks = current_detections_to_tracks
            ## NOTE: AH 8-31-2016: A PROBLEM IS CREATED WHEN MULTIPLE DETECTIONS ARE REGISTERED IN THE SAME FRAME AND A CURRENT 
            ## TRACK IS MISASSIGNED TO A DETECTION THAT SHOULD NOT BE PART OF THAT TRACK. THIS IS LIKELY TO HAPPEN WHEN THE FISH ARE MOVING
            ## QUICKLY. THESE TRACKS WON'T BE LINKED IN THE FOLLOWING STEP BECAUSE THEY CONTAIN A >= 1 FRAME WHERE BOTH TRACKS ARE DETECTED.
        
        self.tracks = tracks
    

    def tracklet_relinking(self):
        last_frames  = map(lambda j: np.max(self.tracks[j][:,0]), range(len(self.tracks)))
        start_frames = map(lambda j: np.min(self.tracks[j][:,0]), range(len(self.tracks)))
        
        relinking_costs  = []
        for i in range(len(self.tracks)):
            track = self.tracks[i]
            last_frame = last_frames[i]
            # find candidates with start times after final time of track,
            # ensure they are within the time window specified by max_lag
            candidates = np.where((start_frames > last_frame) & ((start_frames - last_frame) < self.params["max-lag"]))[0].tolist()
            # construct position array
            d = []
            for j in candidates:
                candidate = self.tracks[j]
                if len(d) == 0:
                    d = candidate[0,[1,2]]
                else:
                    d = np.vstack([d,candidate[0,[1,2]]])
            last_position       = track[track.shape[0]-1, [1,2]]
            candidate_positions = d
            cost = np.array([self.params["max-cost"]] * len(self.tracks))
            if len(candidates) > 0:
                dist = euclidean_distances(last_position, candidate_positions)
                cost[candidates] = dist
            if len(relinking_costs) == 0:
                relinking_costs = cost
            else:
                relinking_costs = np.vstack([relinking_costs, cost])
        
        # AH 8-29: WARNING: PROBLEM SOMEWHERE IN HERE
        # if there is more than one track, try relinking
        if len(self.tracks) > 1:
            relinking_assignment = find_assignment(self.munkres, relinking_costs)
            # re-sequencer
            reassignments = [-1]*len(self.tracks)
            new_tracks    = []
            order         = np.argsort(last_frames)
            # re-sequence in sorted order
            for i in order:
                j, k, cost = relinking_assignment[i,:]
                j = int(j)
                k = int(k)
                if reassignments[i] == -1:
                    reassignments[i] = len(new_tracks)
                    new_tracks.append(self.tracks[i])
                else:
                    new_tracks[reassignments[i]] = np.vstack([
                        new_tracks[reassignments[i]],
                        self.tracks[i]
                    ])
                if cost < self.params["relinking-threshold"]:
                    reassignments[k] = reassignments[i]
            self.tracks = new_tracks
    

    def save(self, tracks_filename):
        numtracks  = len(self.tracks)
        X          = np.zeros((self.numframes, numtracks), dtype='float32')
        Y          = np.zeros((self.numframes, numtracks), dtype='float32')
        framestamp = np.zeros((self.numframes, numtracks), dtype='float32')
        det        = np.zeros((self.numframes, numtracks), dtype='uint8')
        
        for i in range(len(self.tracks)):
            track            = self.tracks[i]
            ts               = track[:,0].astype(int)
            X[ts,i]          = track[:,1]
            Y[ts,i]          = track[:,2]
            framestamp[ts,i] = track[:,3]
            det[ts,i]        = np.ones(len(ts))
        
        f = h5py.File(tracks_filename, 'w')
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("frame", data=framestamp)
        f.create_dataset("det", data=det)
        f.close()

