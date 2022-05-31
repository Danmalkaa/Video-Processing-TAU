from video_stabilization import stabilize_video
from background_subtraction import background_subtraction
from matting import video_matting
from tracking import track_video
import cv2
import logging
import sys

import argparse

parser = argparse.ArgumentParser(
    description="Please pass NO arguments for a full run (or -all flag), or select relevant flags to run specified parts of project.")
parser.add_argument('-all', action='store_true', help='Pass this flag to run entire project process')
parser.add_argument('-video_stab', action='store_true', help='Pass this flag to run Video Stab. part')
parser.add_argument('-bs', action='store_true', help='Pass this flag to run Background Sub. part')
parser.add_argument('-matting', action='store_true', help='Pass this flag to run Matting part')
parser.add_argument('-tracking', action='store_true', help='Pass this flag to run Tracking part')

args = parser.parse_args()
# RUN_ALL = False
if args.all or len(sys.argv) == 1:
    RUN_ALL = True

LOG_FILENAME = '../Outputs/RunTimeLog.txt'
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    filename=LOG_FILENAME,
                    filemode='w',
                    level=logging.INFO)

if RUN_ALL or args.video_stab:
    stabilize_video('../Input/INPUT.avi')
if RUN_ALL or args.bs:
    background_subtraction('../Outputs/stabilize.avi')
if RUN_ALL or args.matting:
    video_matting('../Outputs/stabilize.avi', '../Outputs/binary.avi', cv2.imread('../Input/background.jpg'))
if RUN_ALL or args.tracking:
    track_video('../Outputs/matted.avi')
