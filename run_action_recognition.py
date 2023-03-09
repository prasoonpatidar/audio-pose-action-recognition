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
from sensing.oakdlite.run import pose_model_config, create_vpu_pipeline, create_keypoint_frame
from sensing.audio.run_audio import ReaderThread as audioReaderThread, Device as audioDevice

from otc_models import get_model
from otc_models.yamnet import audio_inference
from otc_models.posec3d import pose_inference
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
    pipeline, stereo = create_vpu_pipeline()
    logger.info("Created VPU Pipeline")

    # get output queue for rgb info and depth info
    depth_recorder_queue = queue.Queue()
    rgb_recorder_queue = queue.Queue()
    pose_recorder_queue = queue.Queue()

    # initialize depth_recorder thread and rgb_recorder thread
    depth_recorder_thread = DepthRecorderThread(depth_recorder_queue, logger, experiment_dir, Config)
    depth_recorder_thread.start()
    logger.info("Initialized Depth Recorder Thread")

    rgb_recorder_thread = RGBRecorderThread(rgb_recorder_queue, logger, experiment_dir, Config)
    rgb_recorder_thread.start()
    logger.info("Initialized RGB Recorder Thread")

    pose_recorder_thread = PoseRecorderThread(pose_recorder_queue, logger, experiment_dir, Config)
    pose_recorder_thread.start()
    logger.info("Initialized Pose Recorder Thread")

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    if visualize:
        rgbWindowName = "rgb"
        depthWindowName = "depth"
        poseWindowName = 'pose'
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(depthWindowName)
        cv2.namedWindow(poseWindowName)

    # mapping OpenPose keypoints to PoseNet
    pose_estimator = get_poseestimator(pose_model_config, **{"decoder": None})
    num_rgb_frames = 0.
    curr_time_rgb = time.time()
    num_depth_frames = 0.
    curr_time_depth = time.time()
    num_pose_frames = 0.
    curr_time_pose = time.time()
    curr_keypoints = None
    checkpoint = time.time()
    ckpt_file = '/tmp/oakdlite.ckpt'

    # setup audio device

    # initialize queues
    sensor_queue = Queue()
    if visualize:
        viz_queue = Queue()
    else:
        viz_queue = None

    # initialize audio device
    audioSensor = audioDevice(run_config, sensor_queue, logger, viz_queue)
    # check if available
    if audioSensor.is_available():
        logger.info(f"- Found Sensor {audioSensor.name}-")
        audioSensor.startReader()
    else:
        print("Audio sensor not available, exiting...")
    # audio_frames = np.zeros((100, audioSensor.num_channels))
    audio_frames = np.zeros((100,1))
    audio_window_name = 'Audio FFt'
    screen_width, screen_height = get_screen_size()
    cv2.namedWindow(audio_window_name)
    cv2.moveWindow(audio_window_name, screen_width // 2, 0)

    # initialize action recognition models

    # get pose and audio based inferences
    pose_otc_model_names = ['posec3d_ntu120', 'posec3d_ntu60', 'posec3d_hmdb', 'posec3d_ucf', 'stgcn_ntu60']
    pose_otc_models = {xr: get_model(xr) for xr in pose_otc_model_names[0:1]}
    pose_otc_queue = Queue()
    pose_curr_time = time.time()
    pose_activity_interval = 10
    pose_activity = 'None'

    audio_otc_model_names = ['yamnet']
    audio_otc_models = {xr: get_model(xr) for xr in audio_otc_model_names}
    audio_otc_queue = Queue()
    audio_curr_time = time.time()
    audio_activity_interval = 2
    audio_activity = 'None'

    pose_instance_data = []
    audio_instance_data = []
    # start running VPU pipeline
    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:
            logger.info("Oakdlite device found, starting processing")
            frameRgb = None
            frameDisp = None

            while True:
                latestPacket = {}
                latestPacket["rgb"] = None
                latestPacket["disp"] = None
                latestPacket["pose"] = None

                queueEvents = device.getQueueEvents(("rgb", "disp", "pose"))
                if time.time() - checkpoint > CHECKPOINT_FREQ:
                    with open(ckpt_file, 'w') as ckpt_f:
                        ckpt_f.write(f'{datetime.datetime.now()}')
                    checkpoint = time.time()
                for queueName in queueEvents:
                    packets = device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        # logger.info(f"Queue packets: {len(packets)}")
                        latestPacket[queueName] = packets[-1]

                frame_time = time.time_ns()
                if latestPacket["rgb"] is not None:
                    # send rgb frame
                    frameRgb = latestPacket["rgb"].getCvFrame()
                    frameRgb = cv2.putText(frameRgb, pose_activity, (0, frameRgb.shape[0] // 2),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # rgb_recorder_queue.put((frame_time, frameRgb))
                    num_rgb_frames += 1
                    if time.time() - curr_time_rgb > 10.:
                        logger.info(f"RGB Frames({frameRgb.shape}) in 10 secs:{num_rgb_frames}")
                        curr_time_rgb = time.time()
                        num_rgb_frames = 0.
                    if visualize:
                        cv2.imshow(rgbWindowName, frameRgb)

                if latestPacket["disp"] is not None:
                    # logger.info("Put Object in queue")
                    frameDisp = latestPacket["disp"].getFrame()
                    # maxDisparity = stereo.initialConfig.getMaxDisparity()
                    # # Optional, extend range 0..95 -> 0..255, for a better visualisation
                    # if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
                    # # Optional, apply false colorization
                    # if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
                    # frameDisp = np.ascontiguousarray(frameDisp)
                    # if visualize:
                    #     cv2.imshow(depthWindowName, frameDisp)
                    # depth_recorder_queue.put((frame_time, frameDisp))
                    # num_depth_frames += 1
                    # if time.time() - curr_time_depth > 10.:
                    #     logger.info(f"Depth Frames({frameDisp.shape}) in 10 secs:{num_depth_frames}")
                    #     curr_time_depth = time.time()
                    #     num_depth_frames = 0.

                if latestPacket["pose"] is not None:
                    nn_out = latestPacket["pose"]
                    keypoints = pose_estimator.get_pose_data(nn_out)
                    if keypoints.shape[0]>0.:
                        pose_instance_data.append((frame_time,keypoints))
                    # curr_keypoints = keypoints
                    # pose_recorder_queue.put((frame_time, keypoints))
                    # if visualize:
                    #     keypointFrame = create_keypoint_frame(keypoints, pose_model_config['input_size'][0],
                    #                                           pose_model_config['input_size'][1], frameRgb.shape[1],
                    #                                           frameRgb.shape[0], resize_img=False)
                    #     cv2.imshow(poseWindowName, keypointFrame)
                    if time.time() > pose_curr_time + pose_activity_interval:
                        for pose_model_name in pose_otc_models:
                            pose_th = threading.Thread(target=pose_inference,
                                                       args=(pose_instance_data,
                                                             pose_otc_models[pose_model_name][0],
                                                             pose_otc_models[pose_model_name][1], pose_otc_queue))
                            pose_th.start()
                            pose_curr_time = time.time()
                            pose_instance_data = list()
                            pass
                        if pose_otc_queue.qsize() > 0:
                            pose_activity = [pose_otc_queue.get() for _ in range(pose_otc_queue.qsize())][-1]
                    num_pose_frames += 1
                    if time.time() - curr_time_pose > 10.:
                        logger.info(f"Pose Frames({keypoints.shape}) in 10 secs:{num_pose_frames}")
                        curr_time_pose = time.time()
                        num_pose_frames = 0.

                if viz_queue.qsize() > 0:
                    # self.logger.info(f"Running viz data from {AUDIO_DEVICE_NAME} sensor...")
                    audio_frame = np.concatenate([viz_queue.get() for _ in range(viz_queue.qsize())])
                    audio_instance_data.append(audio_frame)
                    if time.time() > audio_curr_time + audio_activity_interval:
                        for audio_model_name in audio_otc_models:
                            audio_th = threading.Thread(target=audio_inference,
                                                        args=((np.concatenate(audio_instance_data),
                                                               audioSensor.sampling_rate),
                                                              audio_otc_models[audio_model_name][0],
                                                              audio_otc_models[audio_model_name][1], audio_otc_queue))
                            audio_th.start()
                            audio_curr_time = time.time()
                            audio_instance_data = list()
                            pass
                    if audio_otc_queue.qsize() > 0:
                        audio_activity = [audio_otc_queue.get() for _ in range(audio_otc_queue.qsize())][-1]
                    audio_frames = np.concatenate([audio_frames, audio_frame])
                    audio_frames = audio_frames[-40000:]
                    S_fft = np.abs(librosa.stft(y=audio_frames.T, n_fft=256))
                    S_dB = librosa.amplitude_to_db(S_fft, ref=np.min).mean(axis=0)
                    img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_JET)
                    img_col = cv2.putText(img_col, audio_activity, (0, img_col.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    cv2.imshow(audio_window_name, img_col)
                    # if cv2.waitKey(1) == 27:
                    #     break  # esc to quit
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                    if cv2.getWindowProperty(audio_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if visualize:
                    if cv2.waitKey(1) == 27:
                        break  # esc to quit
                if cv2.getWindowProperty('rgb', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.getWindowProperty('depth', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.getWindowProperty('pose', cv2.WND_PROP_VISIBLE) < 1:
                    break
                if cv2.getWindowProperty(audio_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

            logger.info(f"Exiting gracefully...")
            pose_recorder_queue.put((None, None))
            depth_recorder_queue.put((None, None))
            rgb_recorder_queue.put((None, None))
    except Exception as e:
        logger.info(f"Exiting with error...")
        logger.info(traceback.print_exc())
        pose_recorder_queue.put((None, None))
        depth_recorder_queue.put((None, None))
        rgb_recorder_queue.put((None, None))
        audioSensor.stopWriter()
        audioSensor.stopReader()
        cv2.destroyWindow(audio_window_name)
        logger.info(f"Stopped {audioSensor.name}")
    finally:
        pose_recorder_queue.put((None, None))
        depth_recorder_queue.put((None, None))
        rgb_recorder_queue.put((None, None))
        pose_recorder_thread.stop()
        depth_recorder_thread.stop()
        rgb_recorder_thread.stop()
        pose_recorder_thread.join()
        logger.info("Pose thread joined")
        depth_recorder_thread.join()
        logger.info("Depth thread joined")
        rgb_recorder_thread.join()
        logger.info("RGB thread joined")
        cv2.destroyAllWindows()
        logger.info(f"Oakdlite Sensor closed successfully")
        audioSensor.stopWriter()
        audioSensor.stopReader()
        logger.info(f"Audio Collection Complete {audioSensor.name}")
