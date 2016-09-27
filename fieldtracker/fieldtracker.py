#!/usr/local/bin/python

'''
    fieldtracker
    ~~~~~~
    
    Simple detection and tracking with python and OpenCV.
    Developed by Andrew Hein and Colin Twomey.

    Open Source under a BSD 3-clause License (see LICENSE).
'''

__version__ = "0.1.0"

import sys
import argparse
import ConfigParser

import detector as dt
import linker as ln


def main():
    # Handle command line arguments
    parser = argparse.ArgumentParser(
        description='Run detections and linking on a given video.'
    )
    parser.add_argument(
        "-v", "--version", action="version", help="Version information.",
        version="fieldtracker v{version}".format(version=__version__)
    )
    parser.add_argument(
        "-c", "--config", dest="config", help="Configuration file.", default=None
    )
    parser.add_argument(
        "-g", "--gui", dest="gui", help="Display live output (processed or original).", default=None
    )
    parser.add_argument(
        "-p", "--result-prefix", dest="prefix", help="Path and/or file name prefix for result files.", default=''
    )
    parser.add_argument(
        "-s", "--save-detection-video", dest="det_video", help="Save video of detections.", default=None
    )
    parser.add_argument(
        "-d", "--detections-only", dest="only_detections",
        help="Only run detections.", default=False, action='store_true'
    )
    parser.add_argument(
        "-l", "--linking-only", dest="only_linker",
        help="Only run linker on detections.", default=False, action='store_true'
    )
    parser.add_argument("video", help="Video to track.")
    args = parser.parse_args()

    # Parameter defaults
    detection_params = dt.Detector.parameter_defaults
    linking_params   = ln.Linker.parameter_defaults

    # Read in configuration information, otherwise use defaults
    if args.config != None:
        config = ConfigParser.SafeConfigParser()
        config.read(args.config)
        detection_params.update(config.items("Detection", detection_params))
        linking_params.update(config.items("Linking", linking_params))

    # Construct output filenames
    prefix = args.prefix
    prefix = prefix + ('_' if len(prefix) > 0 and prefix[-1] != '/' else '')
    detection_video_filename = prefix + 'detections.mp4'
    tracks_video_filename    = prefix + 'tracks.mp4'
    detection_filename       = prefix + 'detections.csv'
    tracks_filename          = prefix + 'tracks.h5'

    # Run detector & linking as needed
    if not args.only_linker:
        print "running detections..."
        detector = dt.Detector(detection_filename, detection_params)
        dt.run_detections(detector, args.video, detection_video_filename, args.gui)
    if not args.only_detections:
        print "running linker..."
        linker = ln.Linker(linking_params)
        print "  load detections"
        linker.load_detections(detection_filename)
        print "  compute detection assignments"
        linker.detection_assignments()
        print "  build tracklets"
        linker.build_tracklets()
        print "  run tracklet relinking"
        linker.tracklet_relinking()
        print "  save results"
        linker.save(tracks_filename)

