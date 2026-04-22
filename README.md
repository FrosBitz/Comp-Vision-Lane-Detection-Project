# Ultra-Fast Lane Detection — Inference Pipeline

Minimal inference-only pipeline for Ultra-Fast-Lane-Detection-v2 (CurveLanes ResNet-34).
Based on the paper: [Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification](https://arxiv.org/abs/2206.07389).

## Layout
```
inference/
  config.py        CurveLanes res34 config (hardcoded)
  model.py         ParsingNet + ResNet backbone + checkpoint loader
  postprocess.py   pred -> coords, smoothing, ego-lane selection, drawing
  pipeline.py      CLI + LaneDetector class (image / folder / video)
weight/
  curvelanes_res34.pth
test_images/
```

## Install
```
pip install -r requirements.txt
```

## Usage

CLI:
```
python -m inference.pipeline test_images out/ --draw-style ego --lane-color blue
python -m inference.pipeline clip.mp4 out.mp4
python -m inference.pipeline frame.jpg out.jpg
```

Flags: `--weight`, `--device {auto,cpu,cuda,mps}`, `--draw-style {all,ego}`,
`--lane-color {green,blue,red,yellow,white}`, `--width`.

Library:
```python
import cv2
from inference import LaneDetector

det = LaneDetector("weight/curvelanes_res34.pth")
img = cv2.imread("test_images/test01.jpeg")
det.annotate(img, draw_style="ego", lane_color="blue")
cv2.imwrite("out.jpg", img)
```

## Weights
`weight/curvelanes_res34.pth` — CurveLanes ResNet-34 (F1 81.34).
Original: https://drive.google.com/file/d/1O1kPSr85Icl2JbwV3RBlxWZYhLEHo8EN

## Pipeline

```
input image / folder / video
        │
        ▼
┌─────────────────────┐   LaneDetector.__init__
│  build_model()      │  → ParsingNet (ResNet-34 backbone + cls_row / cls_col heads)
│  load_checkpoint()  │  → curvelanes_res34.pth, raises if required weights missing
│  build_transform()  │  → Resize(1000×1600) + ToTensor + ImageNet normalize
└─────────────────────┘
        │
        ▼  per frame
LaneDetector.predict(bgr)
  1. BGR → RGB → PIL
  2. Resize to 1000×1600, bottom-crop to 800×1600 (sky removal via crop_ratio = 0.8)
  3. net(tensor) → { loc_row, exist_row, loc_col, exist_col }
  4. pred2coords:
       • for each of 10 row-anchor lanes: argmax + softmax-weighted refinement
         along the grid dim → (x, y_anchor) points
       • same for 10 col-anchor lanes → (x_anchor, y) points
       • fuse row + col outputs by lane index into a single polyline per lane
       • drop lanes with < 1/6 of the grid marked valid
  returns: list[list[(x, y)]]
        │
        ▼
LaneDetector.annotate(bgr, draw_style, lane_color, width)
  draw_lanes:
    • draw_style == "ego": keep the lane whose bottom-x is closest to center on
                           each side of the image (fall back to nearest overall
                           if one side is empty)
    • smooth_lane: sort points by dominant axis, degree-2 polyfit, extrapolate
                   down to the image bottom when the curve is y-dominated
    • cv2.polylines (antialiased) in the chosen color
  returns: (annotated_bgr, coords)
        │
        ▼
sink: cv2.imwrite (image / folder), VideoWriter (mp4v), or Streamlit UI
```

Entry points:
- **CLI** — `python -m inference.pipeline <input> <output> [flags]` auto-dispatches folder / image / video.
- **Library** — `from inference import LaneDetector; det.annotate(img)`.
- **Web** — `streamlit run frontend/app.py` wraps the same `LaneDetector` with sidebar controls.

## Citation

```bibtex
@InProceedings{qin2020ultra,
  author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
  title = {Ultra Fast Structure-aware Deep Lane Detection},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year = {2020}
}

@ARTICLE{qin2022ultrav2,
  author = {Qin, Zequn and Zhang, Pengyi and Li, Xi},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {Ultra Fast Deep Lane Detection With Hybrid Anchor Driven Ordinal Classification},
  year = {2022},
  volume = {},
  number = {},
  pages = {1-14},
  doi = {10.1109/TPAMI.2022.3182097}
}
```
