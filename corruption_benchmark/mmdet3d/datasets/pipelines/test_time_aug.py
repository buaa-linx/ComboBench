import mmcv
import warnings
from copy import deepcopy
import os
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Compose
from mmdet3d.core.points.base_points import BasePoints
import numpy as np
from PIL import Image
from .Camera_corruptions import Glare, ImageSnow, ImageFog, ImageRain, ImageGaussianNoise, ImageImpulseNoise, ImageUniformNoise
from .Camera_corruptions import ImageBBoxOperation
from .Camera_corruptions import MotionBlurFrontBack, MotionBlurLeftRight
from .Camera_corruptions import MotionObjectFrontBack, MotionObjectLeftRight
from .Camera_corruptions import ImageDark, ImageOcclusion, FrameLoss, ImageOverExposure, ImagePosterization
import torch
from .LiDAR_corruptions import gaussian_lidar, lidar_crosstalk, density_decimation, cluster_cutout, uniform_lidar, impulse_lidar, bbox_gaussian, scale_distortion, shear_distortion, bbox_uniform, bbox_impulse, projection_jitter, scan_skew, sensor_stuck, fog_lidar, rain_lidar, snow_lidar, fov_truncation, beam_omission, inter_sensor_dropout, echo_depletion, rotate_distortion, set_seed

def format_list_float_06(l) -> None:
    for index, value in enumerate(l):
        l[index] = float('%.6f' % value)
    return l

def load_points(pts_filename):
    """Private function to load point clouds data.

    Args:
        pts_filename (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    points = np.fromfile(pts_filename, dtype=np.float32)
    return points

def remove_close(points, radius=1.0):
    """Removes point too close within a certain radius from origin.

    Args:
        points (np.ndarray): Sweep points.
        radius (float): Radius below which points are removed.
            Defaults to 1.0.

    Returns:
        np.ndarray: Points after removing.
    """
    if isinstance(points, np.ndarray):
        points_numpy = points
    elif isinstance(points, BasePoints):
        points_numpy = points.tensor.numpy()
    else:
        raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]

@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self, transforms, img_scale, pts_scale_ratio, flip=False, flip_direction='horizontal', pcd_horizontal_flip=False, pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]
        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)
        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn('flip_direction has no effect when flip is set to False')
        if self.flip and (not any([t['type'] == 'RandomFlip3D' or t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn('flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with                 different scales and flips.
        """
        aug_data = []
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict


@PIPELINES.register_module()
class CorruptionMethods(object):
    """Test-time augmentation with corruptions.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool, optional): Whether apply flip augmentation.
            Defaults to False.
        flip_direction (str | list[str], optional): Flip augmentation
            directions for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool, optional): Whether apply horizontal
            flip augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool, optional): Whether apply vertical flip
            augmentation to point cloud. Defaults to True.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(self, corruption_severity_dict=None, seed=2025):
        if corruption_severity_dict is None:
            self.corruption_severity_dict = {}
        elif isinstance(corruption_severity_dict, dict):
            self.corruption_severity_dict = corruption_severity_dict
        else:
            self.corruption_severity_dict = {}
        self.seed = int(seed)

    @staticmethod
    def _to_pil_images(images):
        """Keep pipeline images in PIL format without changing channel order."""
        return [
            image if isinstance(image, Image.Image) else Image.fromarray(
                np.ascontiguousarray(np.asarray(image).astype(np.uint8))
            )
            for image in images
        ]

    def _merge_augmented_lidar(self, pts, points_aug):
        pts_np = pts.detach().cpu().numpy()
        points_aug = np.asarray(points_aug)
        if points_aug.ndim != 2 or points_aug.shape[1] < 4:
            return pts.clone()
        min_points = min(pts_np.shape[0], points_aug.shape[0])
        merged = points_aug[:min_points, :4]
        if pts_np.shape[1] > 4:
            merged = np.concatenate([merged, pts_np[:min_points, 4:]], axis=1)
        return pts.new_tensor(merged)

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with
                different scales and flips.
        """
        set_seed(self.seed)
        imgs = results['img']
        pts = results['points'].tensor
        lidar2img = results['lidar2image']
        if isinstance(imgs, list):
            if isinstance(imgs[0], Image.Image):
                imgs = [np.array(img) for img in imgs]
        if 'gaussian_noise' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['gaussian_noise']
            self.gaussian_camera = ImageGaussianNoise(severity, seed=self.seed)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.gaussian_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            imgs = image_aug_list
            results['img'] = self._to_pil_images(imgs)
            imgs = [np.array(img) for img in results['img']]
            points_aug = gaussian_lidar(pts.detach().cpu().numpy().copy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'impulse_noise' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['impulse_noise']
            self.impulse_camera = ImageImpulseNoise(severity, seed=self.seed)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.impulse_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            imgs = image_aug_list
            results['img'] = self._to_pil_images(imgs)
            imgs = [np.array(img) for img in results['img']]
            points_aug = impulse_lidar(pts.detach().cpu().numpy().copy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'uniform_noise' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['uniform_noise']
            self.uniform_camera = ImageUniformNoise(severity)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.uniform_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            imgs = image_aug_list
            results['img'] = self._to_pil_images(imgs)
            imgs = [np.array(img) for img in results['img']]
            points_aug = uniform_lidar(pts.detach().cpu().numpy().copy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'glare' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['glare']
            self.glare = Glare(severity)
            '\n                nuscenes:\n                0    CAM_FRONT_RIGHT,\n                1    CAM_FRONT,\n                2    CAM_FRONT_LEFT,\n                3    CAM_BACK,\n                4    CAM_BACK_LEFT,\n                5    CAM_BACK_RIGHT\n                '
            img_1 = imgs[1][:, :, [2, 1, 0]]
            lidar2img_1 = lidar2img[1]
            img_1, points_aug = self.glare(image=img_1, points=pts, lidar2img=lidar2img_1)
            img_1 = img_1[:, :, [2, 1, 0]]
            imgs[1] = img_1
            results['img'] = self._to_pil_images(imgs)
            imgs = [np.array(img) for img in results['img']]
            pts = points_aug
            results['points'].tensor = pts
        if 'snow' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['snow']
            self.snow = ImageSnow(severity, seed=self.seed)
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.snow(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]])
            points_aug = snow_lidar(pts[:, :4].detach().cpu().numpy().copy(), severity)
            points_aug = self._merge_augmented_lidar(pts, points_aug)
            results['img'] = self._to_pil_images(img_aug_list)
            imgs = [np.array(img) for img in results['img']]
            pts = points_aug
            results['points'].tensor = pts
        if 'fog' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['fog']
            self.fog = ImageFog(severity, seed=self.seed)
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.fog(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]])
            points_aug = fog_lidar(pts[:, :4].detach().cpu().numpy().copy(), severity)
            points_aug = self._merge_augmented_lidar(pts, points_aug)
            results['img'] = self._to_pil_images(img_aug_list)
            imgs = [np.array(img) for img in results['img']]
            pts = points_aug
            results['points'].tensor = pts
        if 'rain' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['rain']
            self.rain = ImageRain(severity, seed=self.seed)
            img_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.rain(img_i)
                img_aug_list.append(img_i[:, :, [2, 1, 0]])
            points_aug = rain_lidar(pts[:, :4].detach().cpu().numpy().copy(), severity)
            points_aug = self._merge_augmented_lidar(pts, points_aug)
            results['img'] = self._to_pil_images(img_aug_list)
            imgs = [np.array(img) for img in results['img']]
            pts = points_aug
            results['points'].tensor = pts
        if 'shear_distortion' in self.corruption_severity_dict:
            gt_bboxes_3d = results.get('gt_bboxes_3d')
            if gt_bboxes_3d is None:
                gt_bboxes_3d = results.get('ann_info', {}).get('gt_bboxes_3d')
            if gt_bboxes_3d is not None and len(gt_bboxes_3d.tensor) > 0:
                severity = self.corruption_severity_dict['shear_distortion']
                self.shear_distortion = ImageBBoxOperation(severity)
                bboxes_corners = gt_bboxes_3d.corners
                bboxes_centers = gt_bboxes_3d.center
                img_aug_list = []
                c = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3][self.shear_distortion.severity - 1]
                b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
                transform_matrix = torch.tensor([[1, 0, b], [d, 1, e], [f, 0, 1]]).float()
                for i in range(6):
                    img_i = imgs[i][:, :, [2, 1, 0]]
                    lidar2img_i = lidar2img[i]
                    img_i = self.shear_distortion(image=img_i, bboxes_centers=bboxes_centers, bboxes_corners=bboxes_corners, transform_matrix=transform_matrix, lidar2img=lidar2img_i)
                    img_aug_list.append(img_i[:, :, [2, 1, 0]])
                points_aug = shear_distortion(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
                pts = pts.new_tensor(points_aug)
                results['points'].tensor = pts
                results['img'] = self._to_pil_images(img_aug_list)
                imgs = [np.array(img) for img in results['img']]
        if 'rotate_distortion' in self.corruption_severity_dict:
            gt_bboxes_3d = results.get('gt_bboxes_3d')
            if gt_bboxes_3d is None:
                gt_bboxes_3d = results.get('ann_info', {}).get('gt_bboxes_3d')
            if gt_bboxes_3d is not None and len(gt_bboxes_3d.tensor) > 0:
                severity = self.corruption_severity_dict['rotate_distortion']
                self.rotate_distortion = ImageBBoxOperation(severity)
                bboxes_corners = gt_bboxes_3d.corners
                bboxes_centers = gt_bboxes_3d.center
                image_aug_list = []
                theta_base = [4, 8, 12, 16, 20, 24][self.rotate_distortion.severity - 1]
                theta_degree = np.random.uniform(theta_base - 2, theta_base + 2) * np.random.choice([-1, 1])
                theta = theta_degree / 180 * np.pi
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                transform_matrix = torch.tensor([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]).float()
                for i in range(6):
                    img_i = imgs[i][:, :, [2, 1, 0]]
                    lidar2img_i = lidar2img[i]
                    img_i = self.rotate_distortion(image=img_i, bboxes_centers=bboxes_centers, bboxes_corners=bboxes_corners, transform_matrix=transform_matrix, lidar2img=lidar2img_i)
                    image_aug_list.append(img_i[:, :, [2, 1, 0]])
                points_aug = rotate_distortion(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
                pts = pts.new_tensor(points_aug)
                results['points'].tensor = pts
                results['img'] = self._to_pil_images(image_aug_list)
                imgs = [np.array(img) for img in results['img']]
        if 'scale_distortion' in self.corruption_severity_dict:
            gt_bboxes_3d = results.get('gt_bboxes_3d')
            if gt_bboxes_3d is None:
                gt_bboxes_3d = results.get('ann_info', {}).get('gt_bboxes_3d')
            if gt_bboxes_3d is not None and len(gt_bboxes_3d.tensor) > 0:
                severity = self.corruption_severity_dict['scale_distortion']
                self.scale_distortion = ImageBBoxOperation(severity)
                bboxes_corners = gt_bboxes_3d.corners
                bboxes_centers = gt_bboxes_3d.center
                image_aug_list = []
                c = [0.3, 0.4, 0.5, 0.7, 0.9, 0.11][self.scale_distortion.severity - 1]
                a = b = d = 1
                r = np.random.randint(0, 3)
                t = np.random.choice([-1, 1])
                a += c * t
                b += c * t
                d += c * t
                transform_matrix = torch.tensor([[a, 0, 0], [0, b, 0], [0, 0, d]]).float()
                for i in range(6):
                    img_i = imgs[i][:, :, [2, 1, 0]]
                    lidar2img_i = lidar2img[i]
                    img_i = self.scale_distortion(image=img_i, bboxes_centers=bboxes_centers, bboxes_corners=bboxes_corners, transform_matrix=transform_matrix, lidar2img=lidar2img_i)
                    image_aug_list.append(img_i[:, :, [2, 1, 0]])
                points_aug = scale_distortion(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
                pts = pts.new_tensor(points_aug)
                results['points'].tensor = pts
                results['img'] = self._to_pil_images(image_aug_list)
                imgs = [np.array(img) for img in results['img']]
        if 'motion_object' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['motion_object']
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
            elif 'ann_info' in results and 'gt_bboxes_3d' in results['ann_info']:
                gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
            else:
                gt_bboxes_3d = None
            if gt_bboxes_3d is None or len(gt_bboxes_3d.tensor) == 0:
                pass
            else:
                self.motion_object_sim_frontback = MotionObjectFrontBack(severity=severity, corrput_list=[0.02 * i for i in range(1, 7)])
                self.motion_object_sim_leftright = MotionObjectLeftRight(severity=severity, corrput_list=[0.02 * i for i in range(1, 7)])
                bboxes_corners = gt_bboxes_3d.corners
                bboxes_centers = gt_bboxes_3d.center
                image_aug_list = []
                for i in range(6):
                    img_i = imgs[i][:, :, [2, 1, 0]]
                    lidar2img_i = lidar2img[i]
                    if isinstance(lidar2img_i, np.ndarray):
                        lidar2img_i = torch.from_numpy(lidar2img_i).float()
                    elif not isinstance(lidar2img_i, torch.Tensor):
                        lidar2img_i = torch.tensor(lidar2img_i, dtype=torch.float32)
                    if i % 3 == 0:
                        img_i = self.motion_object_sim_frontback(image=img_i, lidar2img=lidar2img_i, bboxes_corners=bboxes_corners)
                    else:
                        img_i = self.motion_object_sim_leftright(image=img_i, bboxes_corners=bboxes_corners, lidar2img=lidar2img_i)
                    image_aug_list.append(img_i[:, :, [2, 1, 0]])
                results['img'] = self._to_pil_images(image_aug_list)
                imgs = [np.array(img) for img in results['img']]
        if 'gauss_camera' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['gauss_camera']
            self.gauss_camera = ImageGaussianNoise(severity, seed=self.seed)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                print(img_i.dtype)
                img_i = self.gauss_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'impulse_camera' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['impulse_camera']
            self.impulse_camera = ImageImpulseNoise(severity, seed=self.seed)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.impulse_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'uniform_camera' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['uniform_camera']
            self.uniform_camera = ImageUniformNoise(severity)
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.uniform_camera(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'motion_blur' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['motion_blur']
            self.motion_blur_sim_frontback = MotionBlurFrontBack(severity=severity, corrput_list=[0.02 * i for i in range(1, 7)])
            self.motion_blur_sim_leftright = MotionBlurLeftRight(severity=severity, corrput_list=[0.02 * i for i in range(1, 7)])
            image_aug_list = []
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                if i % 3 == 0:
                    img_i = self.motion_blur_sim_frontback(image=img_i)
                else:
                    img_i = self.motion_blur_sim_leftright(image=img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'dark' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['dark']
            self.dark = ImageDark(severity)
            image_aug_list = []
            print('dark')
            for i in range(6):
                img_i = imgs[i][:, :, [2, 1, 0]]
                img_i = self.dark(img_i)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'occlusion' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['occlusion']
            self.occlusion = ImageOcclusion(severity)
            image_aug_list = []
            for i in range(6):
                img_i = self.occlusion(imgs[i][:, :, [2, 1, 0]], i, imgs)
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'frame_loss' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['frame_loss']
            self.frame_loss = FrameLoss(severity)
            image_aug_list = []
            imgs = np.stack([img[:, :, [2, 1, 0]] for img in imgs], axis=0)
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
            imgs = self.frame_loss(imgs)
            image_aug_list = [imgs[i][:, :, [2, 1, 0]] for i in range(imgs.shape[0])]
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'over_exposure' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['over_exposure']
            self.over_exposure = ImageOverExposure(severity)
            image_aug_list = []
            for i in range(6):
                img_i = self.over_exposure(imgs[i])
                image_aug_list.append(img_i)
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'posterization' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['posterization']
            self.posterization = ImagePosterization(severity)
            image_aug_list = []
            for i in range(6):
                img_i = self.posterization(imgs[i][:, :, [2, 1, 0]])
                image_aug_list.append(img_i[:, :, [2, 1, 0]])
            results['img'] = self._to_pil_images(image_aug_list)
            imgs = [np.array(img) for img in results['img']]
        if 'lidar_crosstalk' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['lidar_crosstalk']
            points_aug = lidar_crosstalk(pts.numpy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'density_decimation' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['density_decimation']
            points_aug = density_decimation(pts.numpy(), severity)
            pts = pts.new_tensor(points_aug)
            pass
            results['points'].tensor = pts
        if 'cluster_cutout' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['cluster_cutout']
            points_aug = cluster_cutout(pts.numpy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'bbox_gaussian' in self.corruption_severity_dict:
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
            else:
                gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
            severity = self.corruption_severity_dict['bbox_gaussian']
            points_aug = bbox_gaussian(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'bbox_impulse' in self.corruption_severity_dict:
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
            else:
                gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
            severity = self.corruption_severity_dict['bbox_impulse']
            points_aug = bbox_impulse(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'bbox_uniform' in self.corruption_severity_dict:
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
            else:
                gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
            severity = self.corruption_severity_dict['bbox_uniform']
            points_aug = bbox_uniform(pts.numpy(), severity, gt_bboxes_3d.tensor.numpy())
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'fov_truncation' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['fov_truncation']
            points_aug = fov_truncation(pts.numpy(), severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'beam_omission' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['beam_omission']
            points_aug = beam_omission(pts, severity)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'echo_depletion' in self.corruption_severity_dict:
            if 'gt_bboxes_3d' in results:
                gt_bboxes_3d = results['gt_bboxes_3d']
            elif 'ann_info' in results and 'gt_bboxes_3d' in results['ann_info']:
                gt_bboxes_3d = results['ann_info']['gt_bboxes_3d']
            else:
                gt_bboxes_3d = None
            if gt_bboxes_3d is None or len(gt_bboxes_3d.tensor) == 0:
                pass
            else:
                severity = self.corruption_severity_dict['echo_depletion']
                points_aug = echo_depletion(pts.numpy(), gt_bboxes_3d.tensor.numpy(), severity)
                pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'inter_sensor_dropout' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['inter_sensor_dropout']
            points_aug = inter_sensor_dropout(pts, severity)
            print(11)
            pts = pts.new_tensor(points_aug)
            results['points'].tensor = pts
        if 'projection_jitter' in self.corruption_severity_dict:
            ori_pose = lidar2img
            severity = self.corruption_severity_dict['projection_jitter']
            noise_pose = []
            for i in range(6):
                noise_pose.append(projection_jitter(ori_pose[i], severity))
            results['lidar2img'] = noise_pose
        if 'scan_skew' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['scan_skew']
            if len(results['sweeps_back']) != 0:
                fir_ego_pose = format_list_float_06(results['ego2global_translation'] + results['ego2global_rotation'])
                fir_sen_glo = format_list_float_06(results['lidar2ego_translation'] + results['lidar2ego_rotation'])
                sec_sweeps = results['sweeps_back'][0]
                sec_ego_pose = format_list_float_06(sec_sweeps['ego2global_translation'] + sec_sweeps['ego2global_rotation'])
                sec_sen_glo = format_list_float_06(sec_sweeps['sensor2ego_translation'] + sec_sweeps['sensor2ego_rotation'])
                pc_pose = np.array([fir_ego_pose, fir_sen_glo, sec_ego_pose, sec_sen_glo])
                points_aug = scan_skew(pts[:, :3].numpy(), pc_pose, severity)
                pts[:, :3] = pts.new_tensor(points_aug)
                results['points'].tensor = pts
            else:
                results['points'].tensor = pts
        if 'sensor_stuck' in self.corruption_severity_dict:
            severity = self.corruption_severity_dict['sensor_stuck']
            frame = sensor_stuck(severity)
            random_number = np.random.random()
            image_aug_list = []
            if random_number < 0.5:
                if len(results['cam_sweeps'][0]) == 0:
                    pass
                else:
                    for i in range(6):
                        filename = os.path.join(results.get('data_root', 'data/nuscenes'), results['cam_sweeps'][i][frame - 1]['filename'])
                        img = mmcv.imread(filename)
                        image_aug_list.append(img)
                    results['img'] = self._to_pil_images(image_aug_list)
                    imgs = [np.array(img) for img in results['img']]
            else:
                lidar_info = results['sweeps']
                if len(lidar_info) < frame - 1:
                    pass
                else:
                    while len(lidar_info) <= frame - 1:
                        frame = frame - 1
                    sweep = lidar_info[frame - 1]
                    points_sweep = load_points(sweep['data_path'])
                    points_sweep = np.copy(points_sweep).reshape(-1, 5)
                    points_aug = remove_close(points_sweep)
                    pts = pts.new_tensor(points_aug)
                    results['points'].tensor = pts

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
