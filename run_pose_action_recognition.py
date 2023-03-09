# !/usr/bin/env python3
'''
This is main file to run data collection from depthcam. It collect rgb, and depth information.
Developer: Prasoon Patidar
Created: 5th March, 2022
'''
import datetime

# basic libraries
import queue
import threading
import time
import traceback
import numpy as np
import cv2
import depthai as dai
import os
import json
import sys
import signal
from queue import Queue
import psutil
import librosa

# Custom libraries
from sensing.utils import get_logger, get_screen_size
from sensing.oakdlite.config import Config
from sensing.oakdlite.depth_recorder import DepthRecorderThread
from sensing.oakdlite.rgb_recorder import RGBRecorderThread
from sensing.oakdlite.pose_recorder import PoseRecorderThread
from sensing.oakdlite.poseestimators import get_poseestimator
from sensing.oakdlite.run import pose_model_config, create_pose_pipeline, create_keypoint_frame
from sensing.audio.run_audio import ReaderThread as audioReaderThread, Device as audioDevice

from otc_models import get_model
from otc_models.yamnet import audio_inference
from otc_models.posec3d import pose_inference
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from util import aggregate_ts_scores

CHECKPOINT_FREQ = 60


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)


class Config:
    VIDEO_SRC = ''

    # -----video config for oakdlite-----

    # If set (True), the ColorCamera is downscaled from 1080p to 720p.
    # Otherwise (False), the aligned depth is automatically upscaled to 1080p
    downscaleColor = True

    # video recording framerate
    rgb_fps = 10
    pose_fps = 10
    depth_fps = 20

    # monocamera resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

    # ----- config for depth/rgb recording -----

    # max duration in single video file
    max_duration_per_video = 30

    # video codec
    video_codec = 'XVID'  # for OSX use 'MJPG', for linux 'XVID'


if __name__ == '__main__':

    # initialize logger
    logger = get_logger('av_runner', 'cache/logs/av/')
    logger.info("------------ New Oakdlite Run ------------")
    config_file = 'dc_config.json'
    visualize = True
    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins'] * 60
    t_data_collection_start = datetime.datetime.now()
    start_time = time.time()
    experiment_dir = run_config["experiment_dir"]
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # get VPU pipeline and stereo config
    pipeline = create_pose_pipeline()
    logger.info("Created Pose Pipeline")


    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    if visualize:
        poseWindowName = 'pose'
        cv2.namedWindow(poseWindowName)

    # mapping OpenPose keypoints to PoseNet
    pose_estimator = get_poseestimator(pose_model_config, **{"decoder": None})
    num_pose_frames = 0.
    curr_time_pose = time.time()
    curr_keypoints = None
    checkpoint = time.time()
    ckpt_file = '/tmp/oakdlite.ckpt'


    # initialize action recognition models

    # get pose and audio based inferences
    pose_otc_model_names = ['posec3d_ntu120', 'posec3d_ntu60', 'posec3d_hmdb', 'posec3d_ucf', 'stgcn_ntu60']
    pose_otc_models = {xr: get_model(xr) for xr in pose_otc_model_names[0:1]}

    model = pose_otc_models['posec3d_ntu120'][0]
    cfg = model.cfg
    device = 'cpu'
    # model device
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)


    pose_otc_queue = Queue()
    pose_curr_time = time.time()
    pose_activity_interval = 10
    pose_activity = 'None'


    pose_instance_data = []
    # start running VPU pipeline
    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:
            logger.info("Oakdlite device found, starting processing")

            while True:
                latestPacket = {}
                latestPacket["pose"] = None

                queueEvents = device.getQueueEvents(("pose"))
                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        # logger.info(f"Queue packets: {len(packets)}")
                        latestPacket[queueName] = packets[-1]

                frame_time = time.time_ns()


                if latestPacket["pose"] is not None:
                    nn_out = latestPacket["pose"]
                    keypoints = pose_estimator.get_pose_data(nn_out)
                    if keypoints.shape[0]>0.:
                        pose_instance_data.append((frame_time,keypoints))
                    curr_keypoints = keypoints
                    if visualize:
                        keypointFrame = create_keypoint_frame(keypoints, pose_model_config['input_size'][0],
                                                              pose_model_config['input_size'][1], 1280,768, resize_img=False)
                        keypointFrame = cv2.putText(keypointFrame, pose_activity, (0, keypointFrame.shape[0] // 2),
                                               cv2.FONT_HERSHEY_SIMPLEX,
                                               0.3, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.imshow(poseWindowName, keypointFrame)
                    if time.time() > pose_curr_time + pose_activity_interval:
                        for pose_model_name in pose_otc_models:
                            pose_th = threading.Thread(target=pose_inference,
                                                       args=(pose_instance_data,
                                                             pose_otc_models[pose_model_name][0],
                                                             pose_otc_models[pose_model_name][1],
                                                             test_pipeline, pose_otc_queue))
                            pose_th.start()
                            pose_curr_time = time.time()
                            pose_instance_data = list()
                        if pose_otc_queue.qsize() > 0:
                            pose_activity = [pose_otc_queue.get() for _ in range(pose_otc_queue.qsize())][-1]
                    num_pose_frames += 1
                    if time.time() - curr_time_pose > 10.:
                        logger.info(f"Pose Frames({keypoints.shape}) in 10 secs:{num_pose_frames}")
                        curr_time_pose = time.time()
                        num_pose_frames = 0.


                if visualize:
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                if cv2.getWindowProperty('pose', cv2.WND_PROP_VISIBLE) < 1:
                    break

            logger.info(f"Exiting gracefully...")
    except Exception as e:
        logger.info(f"Exiting with error...")
        logger.info(traceback.print_exc())
    finally:
        cv2.destroyAllWindows()
        logger.info(f"Oakdlite Sensor closed successfully")
