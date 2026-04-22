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

## Citation
If this project is useful for your research or application, please cite:

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
