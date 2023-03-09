'''
This is main server script to handle remote pose based action recognition requests
'''

import datetime

# basic libraries
import queue
import threading
import time
import traceback
import numpy as np
import cv2
import os
import json
import sys
import signal
import base64
import pickle
from flask import Flask, request

# Custom libraries
from sensing.utils import get_logger
from otc_models import get_model
from otc_models.posec3d import pose_inference_server

DEVICE = 'cuda:0'
app = Flask(__name__)

@app.route("/pose_inference",methods=["POST"])
def pose_inference_request():
    request_json = request.get_json()
    keypoints_encoded = request_json["keypoints"]
    keypoints_score_encoded = request_json["keypoints_score"]

    keypoints = pickle.loads(base64.b64decode(keypoints_encoded.encode()))
    keypoint_scores = pickle.loads(base64.b64decode(keypoints_score_encoded.encode()))
    pose_model_name = request_json["pose_model_name"]
    pose_activity_inference = pose_inference_server(
        keypoints,
        keypoint_scores,
        pose_otc_models[pose_model_name][0],
        pose_otc_models[pose_model_name][1],
    )
    print(f"{datetime.datetime.now()} {pose_model_name}-{pose_activity_inference.split('secs')[0]}s")
    response_dict = {
        'pose_activity':pose_activity_inference
    }
    return json.dumps(response_dict)

if __name__=='__main__':
    # initialize logger
    logger = get_logger('pose_server_runner', 'cache/logs/pose_server/')
    logger.info("------------ New pose server ------------")

    # initialize all pose models
    pose_otc_model_names = ['posec3d_ntu120', 'posec3d_hmdb', 'posec3d_ucf']
    pose_otc_model_devices = ['cuda:0', 'cuda:1', 'cuda:1']
    pose_otc_models = {xr: get_model(xr,device=dr) for (xr,dr) in zip(pose_otc_model_names,pose_otc_model_devices)}

    app.run('0.0.0.0',port=9090,threaded=True,debug=False)
