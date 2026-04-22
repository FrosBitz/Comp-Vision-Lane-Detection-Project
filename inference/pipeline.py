import argparse
from pathlib import Path

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from inference.config import Config
from inference.model import build_model, load_checkpoint
from inference.postprocess import draw_lanes, pred2coords


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def select_device(name):
    if name == "auto" or name is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_transform(cfg):
    return T.Compose([
        T.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class LaneDetector:
    def __init__(self, weight_path, device="auto"):
        self.cfg = Config()
        self.device = select_device(device)
        self.net = build_model(self.cfg, self.device)
        load_checkpoint(self.net, weight_path, self.device)
        self.net.eval()
        self.transform = build_transform(self.cfg)

    @torch.no_grad()
    def predict(self, bgr_image):
        h, w = bgr_image.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil)[:, -self.cfg.train_height:, :].unsqueeze(0).to(self.device)
        pred = self.net(tensor)
        return pred2coords(pred, self.cfg.row_anchor, self.cfg.col_anchor, w, h)

    def annotate(self, bgr_image, draw_style="all", lane_color="green", width=4):
        coords = self.predict(bgr_image)
        draw_lanes(bgr_image, coords, draw_style=draw_style, lane_color=lane_color, width=width)
        return bgr_image, coords


def run_folder(detector, input_dir, output_dir, draw_style, lane_color, width):
    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No images in {input_dir}")
    for path in tqdm(images):
        img = cv2.imread(str(path))
        detector.annotate(img, draw_style, lane_color, width)
        cv2.imwrite(str(out_path / path.name), img)


def run_video(detector, input_path, output_path, draw_style, lane_color, width):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for _ in tqdm(range(total)):
            ok, frame = cap.read()
            if not ok:
                break
            detector.annotate(frame, draw_style, lane_color, width)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Lane Detection inference (CurveLanes res34)")
    parser.add_argument("input", help="image file, folder, or video file")
    parser.add_argument("output", help="output path (folder for images, file for video)")
    parser.add_argument("--weight", default="weight/curvelanes_res34.pth")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--draw-style", default="all", choices=["all", "ego"])
    parser.add_argument("--lane-color", default="green", choices=["green", "blue", "red", "yellow", "white"])
    parser.add_argument("--width", type=int, default=4)
    args = parser.parse_args()

    detector = LaneDetector(args.weight, device=args.device)
    src = Path(args.input)

    if src.is_dir():
        run_folder(detector, src, args.output, args.draw_style, args.lane_color, args.width)
    elif src.suffix.lower() in VIDEO_EXTS:
        run_video(detector, src, args.output, args.draw_style, args.lane_color, args.width)
    elif src.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(src))
        detector.annotate(img, args.draw_style, args.lane_color, args.width)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, img)
    else:
        raise ValueError(f"Unsupported input: {src}")


if __name__ == "__main__":
    main()
