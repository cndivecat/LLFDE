#vis.py
import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import os
import math

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  colors='green',
                  text_color='white',
                  thickness=1,
                  font_scale=0.5,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes with labels (optional) on an image. This is a wrapper of
    mmcv.imshow_bboxes.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): ndarray of shape (k, 4), each row is a bbox in
            format [x1, y1, x2, y2].
        labels (str or list[str], optional): labels of each bbox.
        colors (list[str or tuple or :obj:`Color`]): A list of colors.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    # adapt to mmcv.imshow_bboxes input format
    bboxes = np.split(
        bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [mmcv.color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    img = mmcv.imshow_bboxes(
        img,
        bboxes,
        colors,
        top_k=-1,
        thickness=thickness,
        show=False,
        out_file=None)

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels for _ in range(len(bboxes))]
        assert len(labels) == len(bboxes)

        for bbox, label, color in zip(bboxes, labels, colors):
            if label is None:
                continue
            bbox_int = bbox[0, :4].astype(np.int32)
            # roughly estimate the proper font size
            text_size, text_baseline = cv2.getTextSize(label,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       font_scale, thickness)
            text_x1 = bbox_int[0]
            text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
            text_x2 = bbox_int[0] + text_size[0]
            text_y2 = text_y1 + text_size[1] + text_baseline
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color,
                          cv2.FILLED)
            cv2.putText(img, label, (text_x1, text_y2 - text_baseline),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        mmcv.color_val(text_color), thickness)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img

def bbox_xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0]
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1]

    return bbox_xywh


def bbox_xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0]
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]

    return bbox_xyxy


def bbox_xywh2cs(bbox, aspect_ratio, padding=1., pixel_std=200.):
    """Transform the bbox format from (x,y,w,h) into (center, scale)

    Args:
        bbox (ndarray): Single bbox in (x, y, w, h)
        aspect_ratio (float): The expected bbox aspect ratio (w over h)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale


def bbox_cs2xywh(center, scale, padding=1., pixel_std=200.):
    """Transform the bbox format from (center, scale) to (x,y,w,h). Note that
    this is not an exact inverse operation of ``bbox_xywh2cs`` because the
    normalization of aspect ratio in ``bbox_xywh2cs`` is irreversible.

    Args:
        center (ndarray): Single bbox center in (x, y)
        scale (ndarray): Single bbox scale in (scale_x, scale_y)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.0
        pixel_std (float): The scale normalization factor. Default: 200.0

    Returns:
        ndarray: Single bbox in (x, y, w, h)
    """

    wh = scale / padding * pixel_std
    xy = center - 0.5 * wh
    return np.r_[xy, wh]

def imshow_keypoints(img,
                     pose_result,
                     kpt_score,
                     skeleton=None,
                     kpt_score_thr=0.1,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    
        img = mmcv.imread(img)
        img_h, img_w, _ = img.shape

        for kpts in pose_result:

            kpts = np.array(kpts, copy=False)

            # draw each point on image
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts)

                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord = int(kpt[0]), int(kpt[1])

                    if kpt_score[0][kid] < kpt_score_thr or pose_kpt_color[kid] is None:
                        # skip the point that should not be drawn
                        continue

                    color = tuple(int(c) for c in pose_kpt_color[kid])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

            # draw links
            if skeleton is not None and pose_link_color is not None:
                assert len(pose_link_color) == len(skeleton)

                for sk_id, sk in enumerate(skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                    if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpt_score[0][sk[0]] < kpt_score_thr
                        or kpt_score[0][sk[1]] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                        continue
                    color = tuple(int(c) for c in pose_link_color[sk_id])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                            int(angle), 0, 360, 1)
                        cv2.fillConvexPoly(img_copy, polygon, color)
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(img, pos1, pos2, color, thickness=thickness)

        return img

def imwrite(img, file_path, params=None, auto_mkdir=True):
    
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = osp.abspath(osp.dirname(file_path))
        mkdir_or_exist(dir_name)
    return cv2.imwrite(file_path, img, params)