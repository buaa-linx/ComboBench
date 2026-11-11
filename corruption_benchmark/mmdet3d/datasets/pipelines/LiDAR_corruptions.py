import numpy as np
import random

import imgaug
import torch
from mmdet3d.core.points import BasePoints
from mmdet3d.core.bbox import box_np_ops
from .util.weather import fog_sim, lisa
from .util.weather.fog_sim import simulate_fog, ParameterSet


def rain_lidar(pointcloud, severity):
    rain = lisa.LISA(show_progressbar=False)
    c = [0.2, 0.73, 1.5625, 3.125, 7.29, 10.42][severity - 1]
    points = rain.augment(pointcloud, c)
    return points


def snow_lidar(pointcloud, severity=5):
    snow = lisa.LISA(mode='gunn', show_progressbar=False)
    c = [0.2, 0.73, 1.5625, 3.125, 7.29, 10.42][severity - 1]
    points = snow.augment(pointcloud, c)
    return points


def fog_lidar(pointcloud, severity):
    c = [0.005, 0.01, 0.02, 0.03, 0.06, 0.07][severity - 1]
    parameter_set = ParameterSet(alpha=c, gamma=1e-06)
    points, _, _ = simulate_fog(parameter_set, pointcloud, 1)
    return points.astype(np.float32)


def lidar_crosstalk(pointcloud, severity):
    N, C = pointcloud.shape
    c = [int(0.004 * N), int(0.008 * N), int(0.012 * N), int(0.016 * N), int(0.02 * N), int(0.024 * N)][severity - 1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0
    return pointcloud


def density_decimation(pointcloud, severity):
    N, C = pointcloud.shape
    num = int(N * 0.3)
    c = [int(0.2 * num), int(0.4 * num), int(0.6 * num), int(0.8 * num), num, int(1.2 * num)][severity - 1]
    idx = np.random.choice(N, c, replace=False)
    pointcloud = np.delete(pointcloud, idx, axis=0)
    return pointcloud


def cluster_cutout(pointcloud, severity):
    N, C = pointcloud.shape
    num = int(N * 0.02)
    c = [(2, num), (3, num), (5, num), (7, num), (10, num), (12, num)][severity - 1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    return pointcloud


def gaussian_lidar(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12][severity - 1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def uniform_lidar(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12][severity - 1]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return new_pc


def impulse_lidar(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10, N // 5][severity - 1]
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return pointcloud


def fov_truncation(points, severity):
    angle1 = [-105, -90, -75, -60, -45, -30][severity - 1]
    angle2 = [105, 90, 75, 60, 45, 30][severity - 1]
    if isinstance(points, np.ndarray):
        pts_npy = points
    elif isinstance(points, BasePoints):
        pts_npy = points.tensor.numpy()
    else:
        raise NotImplementedError
    pts_p = (np.arctan(pts_npy[:, 0] / pts_npy[:, 1]) + (pts_npy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2)
    pts_p[pts_p > np.pi] -= np.pi * 2
    pts_p = pts_p / np.pi * 180
    assert np.all(-180 <= pts_p) and np.all(pts_p <= 180)
    filt = np.logical_and(pts_p >= angle1, pts_p <= angle2)
    return points[filt]

def scan_skew(pointcloud, pc_pose, severity):
    from .util.lidar_split import lidar_split, reconstruct_pc_cpu
    ct = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12][severity - 1]
    cr = [0.002, 0.004, 0.006, 0.008, 0.01, 0.12][severity - 1]
    new_pose_list, new_lidar_list = lidar_split(pointcloud, pc_pose)
    r_noise = np.random.normal(size=(100, 3, 3)) * cr
    t_noise = np.random.normal(size=(100, 3)) * ct
    new_pose_list[:, :3, :3] += r_noise
    new_pose_list[:, :3, 3] += t_noise
    f_pc = reconstruct_pc_cpu(new_lidar_list, new_pose_list)
    return f_pc

def bbox_gaussian(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import bbox_gaussian
    pointcloud = pick_bbox(bbox_gaussian, severity, bbox, pointcloud)
    return pointcloud

def bbox_uniform(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import bbox_uniform
    pointcloud = pick_bbox(bbox_uniform, severity, bbox, pointcloud)
    return pointcloud

def bbox_impulse(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import bbox_impulse
    pointcloud = pick_bbox(bbox_impulse, severity, bbox, pointcloud)
    return pointcloud

def scale_distortion(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import scale_distortion
    pointcloud = pick_bbox(scale_distortion, severity, bbox, pointcloud, requires_box=True)
    return pointcloud

def shear_distortion(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import shear_distortion
    pointcloud = pick_bbox(shear_distortion, severity, bbox, pointcloud, requires_box=True)
    return pointcloud

def rotate_distortion(pointcloud, severity, bbox):
    from .util.bbox import pick_bbox
    from .util.bbox.operations import rotate_distortion
    pointcloud = pick_bbox(rotate_distortion, severity, bbox, pointcloud, requires_box=True)
    return pointcloud

def projection_jitter(ori_pose, severity):
    ct = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12][severity - 1] * 2
    cr = [0.002, 0.004, 0.006, 0.008, 0.01, 0.12][severity - 1] * 2
    r_noise = np.random.normal(size=(3, 3)) * cr
    t_noise = np.random.normal(size=3) * ct
    ori_pose[:3, :3] += r_noise
    ori_pose[:3, 3] += t_noise
    return ori_pose

def sensor_stuck(severity):
    frame = [2, 4, 6, 8, 10, 12][severity - 1]
    return frame

# HDL-32E vertical firing angles, in degrees (sorted from lowest to highest).
# Reconstruct approximate ring indices from XYZ because the fifth nuScenes
# BEVFusion feature is sweep time lag, not a LiDAR ring index.
_NUSCENES_HDL32E_VERTICAL_ANGLES = np.array([
    -30.67, -29.33, -28.00, -26.66, -25.33, -24.00, -22.67, -21.33,
    -20.00, -18.67, -17.33, -16.00, -14.67, -13.33, -12.00, -10.67,
    -9.33, -8.00, -6.66, -5.33, -4.00, -2.67, -1.33, 0.00,
    1.33, 2.67, 4.00, 5.33, 6.67, 8.00, 9.33, 10.67,
], dtype=np.float32)


def _estimate_nuscenes_beam_ids(pointcloud):
    """Approximate HDL-32E beam indices (0..31) from XYZ elevation.

    Assumes points are expressed approximately in the originating LiDAR frame.
    Motion-compensated historical sweeps in the keyframe coordinate system
    can be assigned imperfectly: exact ring IDs require preserving the raw
    sweep ring index before coordinate transformation.
    """
    xyz = np.asarray(pointcloud)[:, :3]
    elevation = np.rad2deg(np.arctan2(
        xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])
    ))
    return np.abs(
        elevation[:, None] - _NUSCENES_HDL32E_VERTICAL_ANGLES[None, :]
    ).argmin(axis=1)


def beam_omission(pointcloud, severity):
    if hasattr(pointcloud, 'numpy'):
        pointcloud = pointcloud.numpy()
    elif not isinstance(pointcloud, np.ndarray):
        pointcloud = np.asarray(pointcloud)
    num_beam_to_drop = [8, 12, 16, 20, 24, 28][severity - 1]
    beam_id = _estimate_nuscenes_beam_ids(pointcloud)
    drop_range = (np.arange(len(_NUSCENES_HDL32E_VERTICAL_ANGLES))
                  if num_beam_to_drop == 28 else np.arange(2, 29))
    to_drop = np.random.choice(drop_range, num_beam_to_drop, replace=False)
    for id in to_drop:
        points_to_drop = beam_id == id
        pointcloud = np.delete(pointcloud, points_to_drop, axis=0)
        beam_id = np.delete(beam_id, points_to_drop, axis=0)
    return pointcloud

def inter_sensor_dropout(pointcloud, severity):
    """Drop 8/12/16/20/24/28 beams, then keep every other point."""
    if hasattr(pointcloud, 'numpy'):
        pointcloud = pointcloud.numpy()
    elif not isinstance(pointcloud, np.ndarray):
        pointcloud = np.asarray(pointcloud)
    beam_id = _estimate_nuscenes_beam_ids(pointcloud)
    num_beam_to_drop = [8, 12, 16, 20, 24, 28][severity - 1]
    if num_beam_to_drop == 8:
        to_drop = np.arange(1, 32, 4)
        assert len(to_drop) == 8
    elif num_beam_to_drop == 12:
        to_drop = np.arange(1, 32, 2.67)
        to_drop = to_drop.astype(int)
        assert len(to_drop) == 12
    elif num_beam_to_drop == 16:
        to_drop = np.arange(1, 32, 2)
        assert len(to_drop) == 16
    elif num_beam_to_drop == 20:
        to_drop = np.arange(1, 32, 1.6)
        to_drop = to_drop.astype(int)
        assert len(to_drop) == 20
    elif num_beam_to_drop == 24:
        to_drop = np.arange(1, 32, 1.33)
        to_drop = to_drop.astype(int)
        assert len(to_drop) == 24
    elif num_beam_to_drop == 28:
        to_drop = np.linspace(1, 32, 28, endpoint=False).astype(int)
    for id in to_drop:
        points_to_drop = beam_id == id
        pointcloud = np.delete(pointcloud, points_to_drop, axis=0)
        beam_id = np.delete(beam_id, points_to_drop, axis=0)
    return pointcloud[::2, :]


def echo_depletion(points, boxes, severity):
    drop_rate = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9][severity - 1]
    if np.random.rand() < drop_rate:
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
    return points

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    imgaug.seed(seed)
    fog_sim.RNG = np.random.default_rng(seed)

