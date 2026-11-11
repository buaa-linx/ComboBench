import sys
sys.path.append('utils')
import imgaug.augmenters as iaa
import mmdet3d.datasets.pipelines.util.weather.Automold as am
import torch
import cv2
import os
import numpy as np
import mmcv
import copy
from scipy.ndimage import zoom as scizoom
import PIL
from PIL import Image
import torch.nn.functional as F
from mmdet3d.datasets.pipelines.util.tps_grid_gen import TPSGridGen
import scipy.stats as st
from mmdet3d.core.bbox.structures.utils import points_cam2img

def get_gaussian_kernel(kernlen=5, nsig=3):
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def get_control_point(corners, bboxes_center, lidar2img, transform_matrix, img_height, img_width, imge_changed_flag):
    continue_flag = False
    target_points = None
    source_points = None
    right_line_index = None
    left_line_index = None
    mid_line_index = None
    for ixxxx in [0]:
        clockwise_order_corners = corners[[0, 1, 2, 3, 6, 7, 4, 5]]
        clockwise_order_corners_xy = clockwise_order_corners[[0, 2, 4, 6], :2]
        clockwise_order_corners_theta = safe_arctan_0to2pi(clockwise_order_corners_xy)
        for adjust_step in range(10):
            if clockwise_order_corners_theta.max() - clockwise_order_corners_theta.min() > np.pi:
                clockwise_order_corners_theta[clockwise_order_corners_theta.argmin()] += 2 * np.pi
            else:
                break
        if adjust_step > 3:
            print('error bbox may cover xyz-axis origin')
            continue_flag = True
            continue
        max_theta_idx = clockwise_order_corners_theta.argmax()
        min_theta_idx = clockwise_order_corners_theta.argmin()
        idx_minus = max_theta_idx - min_theta_idx
        if idx_minus.abs() == 1 or idx_minus.abs() == 3:
            only_one_face_saw = True
        else:
            only_one_face_saw = False
        if only_one_face_saw == False:
            clockwise_order_corners_d = clockwise_order_corners_xy.pow(2).sum(1)
            clockwise_order_corners_dmax_index = clockwise_order_corners_d.argmax()
            clockwise_order_corners_xy_near_mask = torch.ones_like(clockwise_order_corners_d).bool().fill_(True)
            clockwise_order_corners_xy_near_mask[clockwise_order_corners_dmax_index] = False
            clockwise_order_corners_theta_near = clockwise_order_corners_theta[clockwise_order_corners_xy_near_mask]
            clockwise_order_corners_theta_near_sort = clockwise_order_corners_theta_near.sort()[0]
            theta_part_length = clockwise_order_corners_theta_near_sort[1:] - clockwise_order_corners_theta_near_sort[:-1]
            if theta_part_length.min() / theta_part_length.sum() < 0.2:
                only_one_face_saw = True
        if only_one_face_saw:
            corners_distence_xy_pow2 = corners[:, :2].pow(2).sum(1)
            nearest_4_corners_indices = (-corners_distence_xy_pow2).topk(k=4)[1]
            nearest_4_corners_xyz = corners[nearest_4_corners_indices]
            nearest_4_corners_uvd = points_cam2img(nearest_4_corners_xyz, proj_mat=lidar2img, with_depth=True)
            nearest_4_corners_uv = nearest_4_corners_uvd[:, :2]
            source_points = nearest_4_corners_uv
            u_valid = (nearest_4_corners_uvd[:, 0] >= 0) * (nearest_4_corners_uvd[:, 0] < img_width)
            v_valid = (nearest_4_corners_uvd[:, 1] >= 0) * (nearest_4_corners_uvd[:, 1] < img_height)
            d_valid = nearest_4_corners_uvd[:, 2] >= 0
            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()
            if not uvd_valid:
                continue_flag = True
                continue
            else:
                imge_changed_flag = True
            nearest_4_corners_xyz_temp = nearest_4_corners_xyz - bboxes_center
            nearest_4_corners_xyz_temp = nearest_4_corners_xyz_temp.mm(transform_matrix)
            target_4_corners_xyz = nearest_4_corners_xyz_temp + bboxes_center
            target_4_corners_uv = points_cam2img(target_4_corners_xyz, proj_mat=lidar2img, with_depth=False)
            target_points = target_4_corners_uv
        else:
            corners_distence_xy_pow2 = corners[:, :2].pow(2).sum(1)
            nearest_6_corners_indices = (-corners_distence_xy_pow2).topk(k=6)[1]
            nearest_6_corners_xyz = corners[nearest_6_corners_indices]
            nearest_6_corners_uvd = points_cam2img(nearest_6_corners_xyz, proj_mat=lidar2img, with_depth=True)
            nearest_6_corners_uv = nearest_6_corners_uvd[:, :2]
            source_points = nearest_6_corners_uv
            u_valid = (nearest_6_corners_uvd[:, 0] >= 0) * (nearest_6_corners_uvd[:, 0] < img_width)
            v_valid = (nearest_6_corners_uvd[:, 1] >= 0) * (nearest_6_corners_uvd[:, 1] < img_height)
            d_valid = nearest_6_corners_uvd[:, 2] >= 0
            uvd_valid = u_valid.all() * v_valid.all() * d_valid.all()
            if not uvd_valid:
                continue_flag = True
                continue
            else:
                imge_changed_flag = True
            nearest_6_corners_xyz_temp = nearest_6_corners_xyz - bboxes_center
            nearest_6_corners_xyz_temp = nearest_6_corners_xyz_temp.mm(transform_matrix)
            target_6_corners_xyz = nearest_6_corners_xyz_temp + bboxes_center
            target_6_corners_uv = points_cam2img(target_6_corners_xyz, proj_mat=lidar2img, with_depth=False)
            target_points = target_6_corners_uv
        if len(source_points) == 4:
            points_u_sort_index = torch.sort(source_points[:, 0])[1]
            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)
            right_line_index = points_u_sort_index[2:4]
            right_line_index = check_order_v(right_line_index, source_points)
        else:
            points_u_sort_index = torch.sort(source_points[:, 0])[1]
            left_line_index = points_u_sort_index[:2]
            left_line_index = check_order_v(left_line_index, source_points)
            mid_line_index = points_u_sort_index[2:4]
            mid_line_index = check_order_v(mid_line_index, source_points)
            right_line_index = points_u_sort_index[4:6]
            right_line_index = check_order_v(right_line_index, source_points)
    return (continue_flag, imge_changed_flag, target_points, source_points, right_line_index, left_line_index, mid_line_index)

def obj_img_transform(image_tensor, target_points, source_points, right_line_index, left_line_index, mid_line_index, smaller_flag):
    total_num_inline = 2
    if len(source_points) == 4:
        source_points_onbox_leftright = get_grid_points(torch.cat([source_points[left_line_index], source_points[right_line_index]]), total_num_inline)
        target_points_onbox_leftright = get_grid_points(torch.cat([target_points[left_line_index], target_points[right_line_index]]), total_num_inline)
    else:
        source_points_onbox_leftmid = get_grid_points(torch.cat([source_points[left_line_index], source_points[mid_line_index]]), total_num_inline)
        source_points_onbox_midright = get_grid_points(torch.cat([source_points[mid_line_index], source_points[right_line_index]]), total_num_inline)
        target_points_onbox_leftmid = get_grid_points(torch.cat([target_points[left_line_index], target_points[mid_line_index]]), total_num_inline)
        target_points_onbox_midright = get_grid_points(torch.cat([target_points[mid_line_index], target_points[right_line_index]]), total_num_inline)
    if len(source_points) == 4:
        source_points_leftright = torch.cat([source_points_onbox_leftright])
        target_points_leftright = torch.cat([target_points_onbox_leftright])
    else:
        source_points_leftmid = torch.cat([source_points_onbox_leftmid])
        source_points_midright = torch.cat([source_points_onbox_midright])
        target_points_leftmid = torch.cat([target_points_onbox_leftmid])
        target_points_midright = torch.cat([target_points_onbox_midright])
    if len(source_points) == 4:
        u1_leftright, v1_leftright = torch.cat([target_points_onbox_leftright, source_points_onbox_leftright]).min(0)[0].long()
        u2_leftright, v2_leftright = torch.cat([target_points_onbox_leftright, source_points_onbox_leftright]).max(0)[0].long()
        image_tensor = patch_transform(u1=u1_leftright, u2=u2_leftright, v1=v1_leftright, v2=v2_leftright, target_points=target_points_leftright, source_points=source_points_leftright, image_tensor=image_tensor, smaller_flag=smaller_flag)
    else:
        u1_leftmid, v1_leftmid = torch.cat([target_points_onbox_leftmid, source_points_onbox_leftmid]).min(0)[0].long()
        u2_leftmid, _ = torch.cat([target_points_onbox_leftmid]).max(0)[0].long()
        _, v2_leftmid = torch.cat([target_points_onbox_leftmid, source_points_onbox_leftmid]).max(0)[0].long()
        image_tensor = patch_transform(u1=u1_leftmid, u2=u2_leftmid, v1=v1_leftmid, v2=v2_leftmid, target_points=target_points_leftmid, source_points=source_points_leftmid, image_tensor=image_tensor, smaller_flag=smaller_flag)
        _, v1_midright = torch.cat([target_points_onbox_midright, source_points_onbox_midright]).min(0)[0].long()
        u1_midright = u2_leftmid
        u2_midright, v2_midright = torch.cat([target_points_onbox_midright, source_points_onbox_midright]).max(0)[0].long()
        image_tensor = patch_transform(u1=u1_midright, u2=u2_midright, v1=v1_midright, v2=v2_midright, target_points=target_points_midright, source_points=source_points_midright, image_tensor=image_tensor, smaller_flag=smaller_flag)
    target_image = image_tensor
    return target_image

def patch_transform(u1, u2, v1, v2, target_points, source_points, image_tensor, smaller_flag):
    try:
        if smaller_flag:
            v1_larger = v1 - (v2 - v1) * 0.5
            v2_larger = v2 + (v2 - v1) * 0.5
            u1_larger = u1 - (u2 - u1) * 0.5
            u2_larger = u2 + (u2 - u1) * 0.5
            v1_larger = v1_larger.int()
            v2_larger = v2_larger.int()
            u1_larger = u1_larger.int()
            u2_larger = u2_larger.int()
            target_height_loacl = v2_larger - v1_larger
            target_width_loacl = u2_larger - u1_larger
            target_points_local = target_points.clone()
            target_points_local[:, 0] -= u1_larger
            target_points_local[:, 1] -= v1_larger
            norm_target_points_local = target_points_local.clone()
            norm_target_points_local[:, 0] = target_points_local[:, 0] * 2 / target_width_loacl - 1
            norm_target_points_local[:, 1] = target_points_local[:, 1] * 2 / target_height_loacl - 1
            source_points_local = source_points.clone()
            source_points_local[:, 0] -= u1_larger
            source_points_local[:, 1] -= v1_larger
            norm_source_points_local = source_points_local.clone()
            norm_source_points_local[:, 0] = source_points_local[:, 0] * 2 / target_width_loacl - 1
            norm_source_points_local[:, 1] = source_points_local[:, 1] * 2 / target_height_loacl - 1
            tps = TPSGridGen(target_height_loacl, target_width_loacl, target_control_points=norm_target_points_local)
            source_coordinate = tps(norm_source_points_local.unsqueeze(0))
            grid_local = source_coordinate.view(1, target_height_loacl, target_width_loacl, 2)
            source_image_local = safe_img_part_get_apply(image=image_tensor, x1=u1_larger, x2=u2_larger, y1=v1_larger, y2=v2_larger)
            target_image_local = F.grid_sample(source_image_local[None], grid_local, padding_mode='border', align_corners=False).squeeze()
            local_center_u1 = int((u2 - u1) * 0.5)
            local_center_v1 = int((v2 - v1) * 0.5)
            target_image_local_center = target_image_local[:, local_center_v1:local_center_v1 + (v2 - v1), local_center_u1:local_center_u1 + (u2 - u1)]
            target_image = safe_img_patch_apply(image=image_tensor, patch=target_image_local_center, x1=u1, y1=v1)
            image_tensor = target_image
        else:
            target_height_loacl = v2 - v1
            target_width_loacl = u2 - u1
            target_points_local = target_points.clone()
            target_points_local[:, 0] -= u1
            target_points_local[:, 1] -= v1
            norm_target_points_local = target_points_local.clone()
            norm_target_points_local[:, 0] = target_points_local[:, 0] * 2 / target_width_loacl - 1
            norm_target_points_local[:, 1] = target_points_local[:, 1] * 2 / target_height_loacl - 1
            source_points_local = source_points.clone()
            source_points_local[:, 0] -= u1
            source_points_local[:, 1] -= v1
            norm_source_points_local = source_points_local.clone()
            norm_source_points_local[:, 0] = source_points_local[:, 0] * 2 / target_width_loacl - 1
            norm_source_points_local[:, 1] = source_points_local[:, 1] * 2 / target_height_loacl - 1
            tps = TPSGridGen(target_height_loacl, target_width_loacl, target_control_points=norm_target_points_local)
            source_coordinate = tps(norm_source_points_local.unsqueeze(0))
            grid_local = source_coordinate.view(1, target_height_loacl, target_width_loacl, 2)
            source_image_local = safe_img_part_get_apply(image=image_tensor, x1=u1, x2=u2, y1=v1, y2=v2)
            target_image_local = F.grid_sample(source_image_local[None], grid_local, padding_mode='border', align_corners=False).squeeze()
            local_center_u1 = int((u2 - u1) * 0.5)
            local_center_v1 = int((v2 - v1) * 0.5)
            target_image_local_center = target_image_local[:,]
            target_image = safe_img_patch_apply(image=image_tensor, patch=target_image_local_center, x1=u1, y1=v1)
            image_tensor = target_image
    except:
        image_tensor = image_tensor
    return image_tensor

def get_4corner(points):
    uv_min = points.min(0)[0]
    uv_max = points.max(0)[0]
    u1, v1 = uv_min
    u2, v2 = uv_max
    return torch.Tensor([[u1, v1], [u1, v2], [u2, v1], [u2, v2]])

def safe_arctan_0to2pi(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    safe_mask = (x != 0) * (y != 0)
    safe_x = x[safe_mask]
    safe_y = y[safe_mask]
    safe_quadrant_0_mask = (safe_x > 0) * (safe_y > 0)
    safe_quadrant_1_mask = (safe_x < 0) * (safe_y > 0)
    safe_quadrant_2_mask = (safe_x < 0) * (safe_y < 0)
    safe_quadrant_3_mask = (safe_x > 0) * (safe_y < 0)
    arctan_value = torch.zeros_like(x)
    safe_arctan_value = safe_quadrant_0_mask * torch.atan(safe_y / safe_x) + safe_quadrant_1_mask * (torch.atan(safe_y / safe_x) + np.pi) + safe_quadrant_2_mask * (torch.atan(safe_y / safe_x) + np.pi) + safe_quadrant_3_mask * (torch.atan(safe_y / safe_x) + 2 * np.pi)
    arctan_value[safe_mask] = safe_arctan_value
    axis_0_mask = (x > 0) * (y == 0)
    axis_1_mask = (x == 0) * (y > 0)
    axis_2_mask = (x < 0) * (y == 0)
    axis_3_mask = (x == 0) * (y < 0)
    origin_mask = (x == 0) * (y == 0)
    arctan_value[origin_mask] = 0
    arctan_value[axis_0_mask] = 0
    arctan_value[axis_1_mask] = np.pi / 2
    arctan_value[axis_2_mask] = np.pi
    arctan_value[axis_3_mask] = np.pi / 2 * 3
    return arctan_value

def check_order_v(index, points):
    if points[index][0, 1] > points[index][1, 1]:
        index_new = index
    else:
        index_new = index[[1, 0]]
    return index_new

def get_grid_points(points4, point_num=4):
    new_points_list = []
    total_line_seg_num = point_num - 1
    for i in range(0, point_num):
        alpha = i / total_line_seg_num
        line_point_a = points4[0] * alpha + points4[1] * (1 - alpha)
        if len(points4) == 2:
            print()
        line_point_b = points4[2] * alpha + points4[3] * (1 - alpha)
        for j in range(0, point_num):
            beta = j / total_line_seg_num
            new_point = line_point_a * beta + line_point_b * (1 - beta)
            new_points_list.append(new_point)
    new_points_tensor = torch.stack(new_points_list)
    return new_points_tensor

def safe_img_patch_apply(image, patch, x1, y1):
    assert len(image.shape) == len(patch.shape)
    assert image.shape[0] == patch.shape[0]
    try:
        w_patch = patch.shape[-1]
        h_patch = patch.shape[-2]
        x2 = x1 + w_patch
        y2 = y1 + h_patch
        w_img = image.shape[-1]
        h_img = image.shape[-2]
        x1_use = x1
        x2_use = x2
        y1_use = y1
        y2_use = y2
        patch_use = patch
        if x1 < 0:
            x1_use = 0
            patch_use = patch_use[:, :, -x1:]
        elif x1 > w_img - 1:
            return image
        if x2 > w_img:
            x2_use = w_img
            patch_use = patch_use[:, :, :-(x2 - w_img)]
        elif x2 < 0:
            return image
        if y1 < 0:
            y1_use = 0
            patch_use = patch_use[:, -y1:, :]
        elif y1 > h_img - 1:
            return image
        if y2 > h_img:
            y2_use = h_img
            patch_use = patch_use[:, :-(y2 - h_img), :]
        elif y2 < 0:
            return image
        image[:, y1_use:y2_use, x1_use:x2_use] = patch_use
    except:
        return image
    return image

def safe_img_part_get_apply(image, x1, x2, y1, y2):
    x1_inverse = 0 - x1
    y1_inverse = 0 - y1
    canvas = image.new_zeros(image.shape[0], y2 - y1, x2 - x1)
    out = safe_img_patch_apply(image=canvas, patch=image, x1=x1_inverse, y1=y1_inverse)
    return out


class Glare:
    def __init__(self, severity) -> None:
        self.severity = severity

    def __call__(self, image, points, lidar2img) -> np.array:
        severity = self.severity
        temp_dict_trans_information = {}
        image_aug = self.image_glare(image, severity, temp_dict_trans_information)
        points_aug = self.lidar_glare(points, lidar2img, severity, temp_dict_trans_information)
        return (image_aug, points_aug)

    def image_glare(self, image, severity, temp_dict_trans_information):
        sun_radius = [30, 40, 50, 60, 70, 80][severity - 1]
        no_of_flare_circles = 1
        src_color = (255, 255, 255)
        img_width = image.shape[1]
        img_height = image.shape[0]
        sun_u_range = [0.25, 0.75]
        sun_v_range = [0.3, 0.45]
        sun_u = np.random.uniform(*sun_u_range) * img_width
        sun_v = np.random.uniform(*sun_v_range) * img_height
        sun_uv = np.array([sun_u, sun_v])
        img_center_coor = np.array([image.shape[1] / 2, image.shape[0] / 2])
        sun_uv_to_center = img_center_coor - sun_uv
        sun_flare_line_angle = np.arctan(sun_uv_to_center[1] / sun_uv_to_center[0])
        image_uint8_rgb = image
        flare_image_rgb = image.astype(np.float64)
        mainflare_mask = np.zeros_like(image).astype(np.float64)
        try:
            flare_image_rgb, mainflare_mask = am.add_sun_flare(image_uint8_rgb, flare_center=sun_uv, angle=sun_flare_line_angle, no_of_flare_circles=no_of_flare_circles, src_radius=sun_radius, src_color=src_color)
        except:
            pass
        flare_image_rgb_uint8 = flare_image_rgb.copy().astype(np.uint8)
        temp_dict_trans_information['glare'] = {}
        temp_dict_trans_information['glare']['mainflare_mask'] = mainflare_mask
        temp_dict_trans_information['glare']['sun_uv'] = sun_uv
        return flare_image_rgb_uint8

    def lidar_glare(self, points, lidar2img, severity, temp_dict_trans_information):
        mainflare_mask = temp_dict_trans_information['glare']['mainflare_mask']
        sun_uv = temp_dict_trans_information['glare']['sun_uv']
        noisy_point_ratio = [0.004, 0.008, 0.012, 0.016, 0.02, 0.024][severity - 1]
        if not (mainflare_mask == 0).all():
            mask_ther = 0.8
            mainflare_mask_lidar = (mainflare_mask[..., 0] > mask_ther * 255.0).astype(np.uint8)
            sun_radis_nonzero = mainflare_mask_lidar.sum(0).nonzero()[0]
            sun_radis = (sun_radis_nonzero.max() - sun_radis_nonzero.min()) / 2
            pts_uvd = points_cam2img(points[:, :3], proj_mat=lidar2img, with_depth=True)
            pts_2d = pts_uvd[:, :2]
            pts_depth = pts_uvd[:, 2]
            pts_keep_flag = ((pts_2d - sun_uv[None]).pow(2).sum(1).sqrt() > sun_radis) + (pts_depth < 0)
            points_keep = points[pts_keep_flag]
        else:
            mask_ther = 0.8
            mainflare_mask_lidar = (mainflare_mask[..., 0] > mask_ther * 255.0).astype(np.uint8)
            pts_uvd = points_cam2img(points[:, :3], proj_mat=lidar2img, with_depth=True)
            pts_2d = pts_uvd[:, :2]
            pts_depth = pts_uvd[:, 2]
            points_keep = points
        points_keep_with_noise = self._lidar_sun_noise(points_keep, noisy_point_ratio)
        return points_keep_with_noise

    def _lidar_sun_noise(self, pointcloud, noisy_point_ratio):
        N, C = pointcloud.shape
        xyz_channel = 3
        noisy_point_num = int(N * noisy_point_ratio)
        index = np.random.choice(N, noisy_point_num, replace=False)
        pointcloud[index, :xyz_channel] += torch.randn(noisy_point_num, xyz_channel) * 3.0
        return pointcloud


class ImageBBoxOperation:
    def __init__(self, severity) -> None:
        self.severity = severity

    def __call__(self, image, lidar2img, transform_matrix, bboxes_centers, bboxes_corners) -> np.array:
        image_tensor_org = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor_org.clone()
        img_height = image_tensor.shape[1]
        img_width = image_tensor.shape[2]
        lidar2img = lidar2img.astype(np.float32)
        bboxes_num = bboxes_corners.shape[0]
        affine_mat = torch.Tensor([[1, 0, 0], [0, 1, 0]])
        bboxes_centers_distance_pow2 = bboxes_centers[:, :2].pow(2).sum(1)
        bboxes_centers_distance_index_far2near = bboxes_centers_distance_pow2.sort(descending=True)[1]
        bboxes_corners = bboxes_corners[bboxes_centers_distance_index_far2near]
        bboxes_centers = bboxes_centers[bboxes_centers_distance_index_far2near]
        imge_changed_flag = False
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]
            bboxes_center = bboxes_centers[idx_b]
            continue_flag, imge_changed_flag, target_points, source_points, right_line_index, left_line_index, mid_line_index = get_control_point(corners, bboxes_center, lidar2img, transform_matrix, img_height, img_width, imge_changed_flag)
            if continue_flag:
                continue
            smaller_flag = False
            if transform_matrix[0, 0] != 0 and transform_matrix[1, 1] != 0 and (transform_matrix[2, 2] != 0) and (transform_matrix[0, 1] == 0) and (transform_matrix[0, 2] == 0) and (transform_matrix[1, 0] == 0) and (transform_matrix[1, 2] == 0) and (transform_matrix[2, 0] == 0) and (transform_matrix[2, 1] == 0):
                if transform_matrix[0, 0] < 1 or transform_matrix[1, 1] < 1 or transform_matrix[2, 2] < 1:
                    smaller_flag = True
            target_image = obj_img_transform(image_tensor, target_points, source_points, right_line_index, left_line_index, mid_line_index, smaller_flag)
            image_tensor = target_image
        if imge_changed_flag:
            image_aug = target_image
            image_aug_np_rgb_255 = (image_aug * 255).permute(1, 2, 0).numpy().astype(np.uint8)
            return image_aug_np_rgb_255
        else:
            return image

class MotionObjectFrontBack:
    def __init__(self, severity, corrput_list=[0.02 * i for i in range(1, 7)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity - 1]

    def __call__(self, image, lidar2img, bboxes_corners) -> np.array:
        img_height = image.shape[0]
        img_width = image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0], canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=lidar2img, with_depth=True)
            corners_uv = corners_uvd[:, :2]
            corners_depth = corners_uvd[:, 2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)
            mask = mask + mask_temp
        mask_bool_float = (mask > 0).astype(np.float32)[:, :, None]
        image_aug_layer = self.zoom_blur(image_rgb_255, corrpution)
        images_aug = image_aug_layer * mask_bool_float + (1 - mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)
        return image_aug_np_rgb_255

    def zoom_blur(self, x, corrpution):
        if corrpution <= 0.02:
            c = np.arange(1, 1 + corrpution, 0.005)
        else:
            c = np.linspace(1, 1 + corrpution, 4)
        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        set_exception = False
        for zoom_factor in c:
            if len(x.shape) < 3 or x.shape[2] < 3:
                x_channels = np.array([x, x, x]).transpose((1, 2, 0))
                zoom_layer = self.clipped_zoom(x_channels, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
            else:
                zoom_layer = self.clipped_zoom(x, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]
            try:
                out += zoom_layer
            except ValueError:
                set_exception = True
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer
        if set_exception:
            print('ValueError for zoom blur, Exception handling')
        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def clipped_zoom(self, img, zoom_factor):
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2
        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1], (zoom_factor, zoom_factor, 1), order=1)
        return img

class MotionObjectLeftRight:

    def __init__(self, severity, corrput_list=[0.02 * i for i in range(1, 7)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity - 1]

    def __call__(self, image, lidar2img, bboxes_corners) -> np.array:

        img_height = image.shape[0]
        img_width = image.shape[1]
        corrpution = self.corrpution
        image_rgb_255 = image
        img_width = image.shape[1]
        kernel_size = corrpution * img_width * 0.5
        kernel_size = int(kernel_size)
        self.iaa_seq = iaa.Sequential([iaa.MotionBlur(k=kernel_size, angle=90)])
        canvas = image.copy()
        bboxes_num = bboxes_corners.shape[0]
        mask = np.zeros((canvas.shape[0], canvas.shape[1]))
        for idx_b in range(bboxes_num):
            corners = bboxes_corners[idx_b]
            mask_temp = np.zeros_like(mask)
            corners_uvd = points_cam2img(corners, proj_mat=lidar2img, with_depth=True)
            corners_uv = corners_uvd[:, :2]
            corners_depth = corners_uvd[:, 2]
            corners_keep_flag = corners_depth > 0
            corners_uv = corners_uv[corners_keep_flag]
            if corners_uv.shape[0] == 0:
                continue
            hull = cv2.convexHull(corners_uv.numpy().astype(np.int))
            cv2.fillConvexPoly(mask_temp, hull, 1)
            mask = mask + mask_temp
        mask_bool_float = (mask > 0).astype(np.float32)[:, :, None]
        images_rgb_255 = image_rgb_255[None]
        image_aug_layer = self.iaa_seq(images=images_rgb_255)[0]
        images_aug = image_aug_layer * mask_bool_float + (1 - mask_bool_float) * image_rgb_255
        image_aug_np_rgb_255 = images_aug.astype(np.uint8)
        return image_aug_np_rgb_255


class MotionBlurFrontBack:
    def __init__(self, severity, corrput_list=[0.02 * i for i in range(1, 7)]) -> None:
        self.severity = severity
        self.corrpution = corrput_list[severity - 1]

    def __call__(self, image) -> np.array:
        corrpution = self.corrpution
        image_rgb_255 = image
        images_aug = self.zoom_blur(image_rgb_255, corrpution)
        image_aug_rgb_255 = images_aug
        return image_aug_rgb_255

    def zoom_blur(self, x, corrpution):
        c = np.arange(1, 1 + corrpution, 0.005)
        x = (np.array(x) / 255.0).astype(np.float32)
        out = np.zeros_like(x)
        set_exception = False
        for zoom_factor in c:
            if len(x.shape) < 3 or x.shape[2] < 3:
                x_channels = np.array([x, x, x]).transpose((1, 2, 0))
                zoom_layer = self.clipped_zoom(x_channels, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], 0]
            else:
                zoom_layer = self.clipped_zoom(x, zoom_factor)
                zoom_layer = zoom_layer[:x.shape[0], :x.shape[1], :]
            try:
                out += zoom_layer
            except ValueError:
                set_exception = True
                out[:zoom_layer.shape[0], :zoom_layer.shape[1]] += zoom_layer
        if set_exception:
            print('ValueError for zoom blur, Exception handling')
        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def clipped_zoom(self, img, zoom_factor):
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2
        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1], (zoom_factor, zoom_factor, 1), order=1)
        return img

class MotionBlurLeftRight:
    def __init__(self, severity, corrput_list=[0.02 * i for i in range(1, 7)]) -> None:
        self.severity = severity
        self.corrput_list = corrput_list

    def __call__(self, image) -> np.array:
        img_width = image.shape[1]
        kernel_size = self.corrput_list[self.severity - 1] * img_width * 0.5
        kernel_size = int(kernel_size)
        self.iaa_seq = iaa.Sequential([iaa.MotionBlur(k=kernel_size, angle=90)])
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        return image_aug_rgb_255

class ImageGaussianNoise:
    def __init__(self, severity, seed=None) -> None:
        self.seed = seed
        self.iaa_seq = iaa.Sequential([iaa.imgcorruptlike.GaussianNoise(severity=severity, seed=self.seed)])

    def __call__(self, image) -> np.array:
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        return image_aug_rgb_255

class ImageImpulseNoise:
    def __init__(self, severity, seed=None) -> None:
        self.seed = seed
        self.iaa_seq = iaa.Sequential([iaa.imgcorruptlike.ImpulseNoise(severity=severity, seed=self.seed)])

    def __call__(self, image) -> np.array:
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        return image_aug_rgb_255

class ImageUniformNoise:
    def __init__(self, severity) -> None:
        self.severity = severity

    def __call__(self, image) -> np.array:
        image_rgb_255 = image
        severity = self.severity
        image_aug_rgb_255 = self.uniform_noise(image_rgb_255, severity)
        return image_aug_rgb_255

    def uniform_noise(self, x, severity):
        c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.0
        return (np.clip(x + np.random.uniform(low=-c, high=c, size=x.shape), 0, 1) * 255).astype(np.uint8)

class ImageSnow:
    def __init__(self, severity, seed=None) -> None:
        self.seed = seed
        self.iaa_seq = iaa.Sequential([iaa.imgcorruptlike.Snow(severity=severity, seed=self.seed)])

    def __call__(self, image) -> np.array:
        images = image[None]
        images_aug = self.iaa_seq(images=images)
        image_aug = images_aug[0]
        gray_ratio = 0.3
        image_aug = gray_ratio * np.ones_like(image_aug) * 128 + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)
        image_rgb_255 = image_aug
        img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] / img_hsv[:, :, 2].max() * 256 * 0.7
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        image_aug = image_rgb_255
        return image_aug

class ImageFog:
    def __init__(self, severity, seed=None) -> None:
        self.seed = seed
        self.iaa_seq = iaa.Sequential([iaa.imgcorruptlike.Fog(severity=severity, seed=self.seed)])
        self.gray_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6][severity - 1]

    def __call__(self, image) -> np.array:
        image_rgb_255 = image
        images = image_rgb_255[None]
        images_aug = self.iaa_seq(images=images)
        image_aug_rgb_255 = images_aug[0]
        image_aug = image_aug_rgb_255
        gray_ratio = self.gray_ratio
        image_aug = gray_ratio * np.ones_like(image_aug) * 128 + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)
        return image_aug

class ImageRain:
    def __init__(self, severity, seed=None) -> None:
        self.seed = seed
        density = [(0.01, 0.06), (0.06, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3)][severity - 1]
        self.iaa_seq = iaa.Sequential([iaa.RainLayer(density=density, density_uniformity=(0.8, 1.0), drop_size=(0.4, 0.6), drop_size_uniformity=(0.2, 0.5), angle=(-15, 15), speed=(0.04, 0.2), blur_sigma_fraction=(0.0001, 0.001), blur_sigma_limits=(0.5, 3.75), seed=self.seed)])

    def __call__(self, image) -> np.array:
        images = image[None]
        images_aug = self.iaa_seq(images=images)
        image_aug = images_aug[0]
        gray_ratio = 0.3
        image_aug = gray_ratio * np.ones_like(image_aug) * 128 + (1 - gray_ratio) * image_aug
        image_aug = image_aug.astype(np.uint8)
        image_rgb_255 = image_aug
        img_hsv = cv2.cvtColor(image_rgb_255, cv2.COLOR_RGB2HSV).astype(np.int64)
        img_hsv[:, :, 2] = img_hsv[:, :, 2] / img_hsv[:, :, 2].max() * 256 * 0.7
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
        img_hsv = img_hsv.astype(np.uint8)
        image_rgb_255 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        image_aug = image_rgb_255
        return image_aug

class ImageOcclusion:
    def __init__(self, severity, mask_file=str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'assets' / 'Occlusion_mask' / 'Mud_Mask_selected'), color_type='unchanged'):
        self.severity = severity
        self.mask_file = mask_file
        self.color_type = color_type
        self.cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']

    def put_mask_on_img(self, img, mask):
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = mask / 255
        alpha = np.power(alpha, 3)
        img_with_mask = alpha * img + (1 - alpha) * mask
        return img_with_mask

    def __call__(self, image, i, imgs):
        if type(imgs) == list and len(imgs) == 6:
            occlusion_id = self.cam_orders[:self.severity + 1]
            if self.cam_orders[i] in occlusion_id:
                mask_id_png = 'mask_' + str(np.random.randint(1, 17)) + '.jpg'
                mask_name = os.path.join(self.mask_file, mask_id_png)
                mask = mmcv.imread(mask_name, self.color_type)
                image_aug = self.put_mask_on_img(image, mask)
            else:
                image_aug = image
        else:
            mask_id_png = 'mask_' + str(np.random.randint(1, 17)) + '.jpg'
            mask_name = os.path.join(self.mask_file, mask_id_png)
            mask = mmcv.imread(mask_name, self.color_type)
            image_aug = self.put_mask_on_img(image, mask)
        return image_aug

class ImageDark:
    def __init__(self, severity):
        self.severity = severity

    def imadjust(self, x, a, b, c, d, gamma=1):
        y = ((x - a) / (b - a)) ** gamma * (d - c) + c
        return y

    def poisson_gaussian_noise(self, x, severity):
        c_poisson = 10 * [60, 25, 12, 5, 3][severity - 1]
        x = np.array(x) / 255.0
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
        c_gauss = 0.1 * [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
        x = np.array(x) / 255.0
        x = np.clip(x + np.random.normal(size=x.shape, scale=c_gauss), 0, 1) * 255
        return np.uint8(x)

    def __call__(self, img):
        c = [0.6, 0.5, 0.4, 0.3, 0.2, 0.15][self.severity - 1]
        img = np.array(img) / 255.0
        img_scaled = self.imadjust(img, img.min(), img.max(), 0, c, gamma=2.0) * 255
        img_scaled = self.poisson_gaussian_noise(img_scaled, severity=self.severity)
        return img_scaled

class FrameLoss:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        img = copy.deepcopy(img)
        assert img.min() >= 0 and img.max() <= 255, 'Image pixel out of range'
        if type(img) == list and len(img) == 6:
            M, C, H, W = img.size()
            for m in range(M):
                prob = np.random.rand()
                if prob < self.severity * 1.0 / 6.0:
                    img[m] = 0
            assert img.min() >= 0 and img.max() <= 255, 'Image pixel out of range'
            return img.permute(0, 2, 3, 1).numpy()
        else:
            M, C, H, W = img.size()
            for m in range(M):
                prob = np.random.rand()
                if prob < self.severity * 1.0 / 6.0:
                    img[m] = 0
            assert img.min() >= 0 and img.max() <= 255, 'Image pixel out of range'
            return img.permute(0, 2, 3, 1).numpy()

class ImageOverExposure:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        exposure_factor = [1.5, 1.7, 2, 2.3, 2.7, 3][self.severity - 1]
        if img is None:
            raise ValueError('Input image is empty')
        if len(img.shape) == 2:
            img = img.astype(np.float32)
            img = np.clip(img * exposure_factor, 0, 255)
        elif len(img.shape) == 3:
            channels = cv2.split(img)
            processed_channels = []
            for channel in channels:
                channel = channel.astype(np.float32)
                channel = np.clip(channel * exposure_factor, 0, 255)
                processed_channels.append(channel)
            img = cv2.merge(processed_channels)
        else:
            raise ValueError('not support image format!')
        return img.astype(np.uint8)

class ImagePosterization:
    def __init__(self, severity):
        self.severity = severity

    def __call__(self, img):
        bits = 6 - self.severity
        img = Image.fromarray(img)
        img = PIL.ImageOps.posterize(img, bits)
        return np.asarray(img)
