import cv2
import numpy as np
import torch


LANE_COLORS = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
}


def pred2coords(pred, row_anchor, col_anchor, image_width, image_height, local_width=1):
    num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape[1:]
    num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape[1:]

    max_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    max_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()
    loc_row = pred["loc_row"].cpu()
    loc_col = pred["loc_col"].cpu()

    row_min_length = num_cls_row / 6
    col_min_length = num_cls_col / 6
    lanes_by_idx = {}

    for i in range(num_lane_row):
        if valid_row[0, :, i].sum() <= row_min_length:
            continue
        pts = []
        for k in range(valid_row.shape[1]):
            if not valid_row[0, k, i]:
                continue
            idx = torch.arange(
                max(0, max_row[0, k, i] - local_width),
                min(num_grid_row - 1, max_row[0, k, i] + local_width) + 1,
            )
            x = (loc_row[0, idx, k, i].softmax(0) * idx.float()).sum() + 0.5
            x = x / (num_grid_row - 1) * image_width
            pts.append((int(x), int(row_anchor[k] * image_height)))
        lanes_by_idx.setdefault(i, []).extend(pts)

    for i in range(num_lane_col):
        if valid_col[0, :, i].sum() <= col_min_length:
            continue
        pts = []
        for k in range(valid_col.shape[1]):
            if not valid_col[0, k, i]:
                continue
            idx = torch.arange(
                max(0, max_col[0, k, i] - local_width),
                min(num_grid_col - 1, max_col[0, k, i] + local_width) + 1,
            )
            y = (loc_col[0, idx, k, i].softmax(0) * idx.float()).sum() + 0.5
            y = y / (num_grid_col - 1) * image_height
            pts.append((int(col_anchor[k] * image_width), int(y)))
        lanes_by_idx.setdefault(i, []).extend(pts)

    return [lanes_by_idx[i] for i in sorted(lanes_by_idx)]


def smooth_lane(lane, image_width, image_height, samples=120, extend_to_bottom=True):
    pts = np.asarray(lane, dtype=np.float32)
    if len(pts) < 3:
        return pts.astype(np.int32)

    x, y = pts[:, 0], pts[:, 1]
    use_y = (y.max() - y.min()) >= (x.max() - x.min())
    axis, values = (y, x) if use_y else (x, y)

    order = np.argsort(axis)
    axis, values = axis[order], values[order]
    unique_axis, unique_idx = np.unique(axis, return_index=True)
    if len(unique_axis) < 3:
        return pts.astype(np.int32)

    values = values[unique_idx]
    degree = min(2, len(unique_axis) - 1)
    try:
        fit = np.poly1d(np.polyfit(unique_axis, values, degree))
    except np.linalg.LinAlgError:
        return pts.astype(np.int32)

    axis_min, axis_max = unique_axis.min(), unique_axis.max()
    if extend_to_bottom and use_y:
        axis_max = max(axis_max, image_height - 1)
    dense_axis = np.linspace(axis_min, axis_max, samples)
    dense_values = fit(dense_axis)
    dense = np.column_stack([dense_values, dense_axis] if use_y else [dense_axis, dense_values])
    dense[:, 0] = np.clip(dense[:, 0], 0, image_width - 1)
    dense[:, 1] = np.clip(dense[:, 1], 0, image_height - 1)
    return dense.astype(np.int32)


def _lane_bottom_x(lane, image_height):
    pts = np.asarray(lane, dtype=np.float32)
    lower = pts[pts[:, 1] >= image_height * 0.55]
    if len(lower) == 0:
        lower = pts
    band = lower[lower[:, 1] >= lower[:, 1].max() - image_height * 0.08]
    if len(band) == 0:
        band = lower
    return float(np.median(band[:, 0]))


def select_ego_lanes(coords, image_width, image_height):
    center = image_width * 0.5
    scored = []
    for lane in coords:
        if len(lane) < 2:
            continue
        bx = _lane_bottom_x(lane, image_height)
        scored.append((abs(bx - center), bx, lane))

    left = [s for s in scored if s[1] < center]
    right = [s for s in scored if s[1] >= center]
    selected = []
    if left:
        selected.append(min(left, key=lambda s: s[0])[2])
    if right:
        selected.append(min(right, key=lambda s: s[0])[2])

    if len(selected) < 2:
        for _, _, lane in sorted(scored, key=lambda s: s[0]):
            if lane not in selected:
                selected.append(lane)
            if len(selected) == 2:
                break
    return selected


def draw_lanes(image, coords, draw_style="all", lane_color="green", width=4):
    h, w = image.shape[:2]
    lanes = select_ego_lanes(coords, w, h) if draw_style == "ego" else coords
    color = LANE_COLORS.get(lane_color, LANE_COLORS["green"])
    for lane in lanes:
        if len(lane) > 1:
            pts = smooth_lane(lane, w, h)
            cv2.polylines(image, [pts], False, color, width, lineType=cv2.LINE_AA)
