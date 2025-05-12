import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
from mmcv import imread
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

detector = init_detector(
    '/home/featurize/work/Image_Website/ethanYaoyx-Image_Generation_for_Medical_Application_Dog_hip/mmdetection/data/rtmdet_m_dog_hip_all.py',
    
    '/home/featurize/work/Image_Website/ethanYaoyx-Image_Generation_for_Medical_Application_Dog_hip/mmdetection/checkpoint/rtmdet_m_dog_hip_287-55485ded.pth',
    device=device
)

pose_estimator = init_pose_estimator(
    '/home/featurize/work/Image_Website/ethanYaoyx-Image_Generation_for_Medical_Application_Dog_hip/mmpose/data/rtmpose-m-Dog_hip_all.py',
    '/home/featurize/work/Image_Website/ethanYaoyx-Image_Generation_for_Medical_Application_Dog_hip/mmpose/checkpoint/all_m_best_PCK_epoch_5-eac12d89_20240725.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': False}}}
)

def calculate_angle(center, head, opposite_center):
    vector1 = np.array(head) - np.array(center)
    vector2 = np.array(opposite_center) - np.array(center)
    cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def predict_keypoints(image_path, conf_thres=0.38, result_image_dir=None, label_save_dir=None):
    image = imread(image_path)
    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))

    detect_result_all = inference_detector(detector, image_path)
    pred_instance_all = detect_result_all.pred_instances.cuda().numpy()

    bboxes_all = np.concatenate((pred_instance_all.bboxes, pred_instance_all.scores[:, None]), axis=1)
    bboxes_head = bboxes_all[np.logical_and(pred_instance_all.labels == 0, pred_instance_all.scores > conf_thres)]
    bboxes_center = bboxes_all[np.logical_and(pred_instance_all.labels == 1, pred_instance_all.scores > conf_thres)]
    bboxes_head = bboxes_head[nms(bboxes_head, 0.3)][:, :4]
    bboxes_center = bboxes_center[nms(bboxes_center, 0.3)][:, :4]

    head, center = [], []
    if len(bboxes_head) > 0:
        pose_results_head = inference_topdown(pose_estimator, image_path, bboxes_head)
        try:
            head = merge_data_samples(pose_results_head).pred_instances.keypoints
        except ValueError:
            pass
    if len(bboxes_center) > 0:
        pose_results_center = inference_topdown(pose_estimator, image_path, bboxes_center)
        try:
            center = merge_data_samples(pose_results_center).pred_instances.keypoints
        except ValueError:
            pass

    predicted_angle1 = predicted_angle2 = predicted_radius1 = predicted_radius2 = None
    mat_data = {}

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    if len(head) >= 2 and len(center) >= 2:
        centers_pred = sorted(center, key=lambda p: p[0, 0])
        heads_pred = sorted(head, key=lambda p: p[0, 0])

        predicted_radius1 = max(bboxes_center[0][2] - bboxes_center[0][0], bboxes_center[0][3] - bboxes_center[0][1]) / 2
        predicted_radius2 = max(bboxes_center[1][2] - bboxes_center[1][0], bboxes_center[1][3] - bboxes_center[1][1]) / 2

        predicted_angle1 = calculate_angle(centers_pred[0][0], heads_pred[0][0], centers_pred[1][0])
        predicted_angle2 = calculate_angle(centers_pred[1][0], heads_pred[1][0], centers_pred[0][0])

        mat_data = {
            'Four_points': np.array([
                centers_pred[0][0],
                centers_pred[1][0],
                heads_pred[0][0],
                heads_pred[1][0],
                [predicted_radius1, predicted_radius2]
            ]),
            'Angles': np.array([[predicted_angle1, predicted_angle2]])
        }
    else:
        # 预测失败也写入空文件
        print('Prediction failed!')
        mat_data = {
            'Four_points': np.zeros((5, 2), dtype=float),
            'Angles': np.array([[0.0, 0.0]])
        }

    if label_save_dir:
        mat_path = os.path.join(label_save_dir, f"{name}.mat")
        sio.savemat(mat_path, mat_data)

    return None, predicted_angle1, predicted_angle2, mat_data



