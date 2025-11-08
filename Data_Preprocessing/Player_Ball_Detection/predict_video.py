import argparse
import os
import cv2
import numpy as np
import csv
import json
import time
import pandas as pd  # used only if you prefer to finalize/convert later; we keep csv module for streaming
import imutils

from Models.tracknet import trackNet
from TrackPlayers.trackplayers import get_output_layers, predict_players

# ----------------------- CLI -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_video_path", type=str, required=True)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--save_weights_path", type=str, required=True)
parser.add_argument("--n_classes", type=int, required=True)
parser.add_argument("--path_yolo_classes", type=str, required=True)
parser.add_argument("--path_yolo_weights", type=str, required=True)
parser.add_argument("--path_yolo_config", type=str, required=True)

# NEW: checkpoint + resume
parser.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint if present and input video matches.")
parser.add_argument("--checkpoint_dir", type=str, default="",
                    help="Where to store the checkpoint json. Defaults next to output.")
parser.add_argument("--commit_every", type=int, default=50,
                    help="Flush CSV & write checkpoint every N frames.")
args = parser.parse_args()

# ------------------- Paths / Outputs -------------------
input_video_path = args.input_video_path
output_video_path = args.output_video_path

if output_video_path == "":
    base_out_dir = os.path.join("VideoOutput")
else:
    base_out_dir = os.path.dirname(output_video_path) or "VideoOutput"
os.makedirs(base_out_dir, exist_ok=True)

frames_dir = os.path.join(base_out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

csv_path = os.path.join(base_out_dir, "tracking_data.csv")

ckpt_dir = args.checkpoint_dir or base_out_dir
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "predict_video.ckpt.json")

# ------------------- Video signature -------------------
def video_signature(path: str):
    ap = os.path.abspath(path)
    try:
        size = os.path.getsize(ap)
        mtime = os.path.getmtime(ap)
    except OSError:
        size, mtime = -1, -1
    return {"path": ap, "size": size, "mtime": mtime}

def load_ckpt(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def save_ckpt(path: str, sig: dict, last_frame: int, total_frames: int, complete: bool = False):
    tmp = path + ".tmp"
    data = {
        "video": sig,
        "last_frame": int(last_frame),
        "total_frames": int(total_frames),
        "complete": bool(complete),
        "timestamp": time.time(),
        "version": 1
    }
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# ------------------- Open video -------------------
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print("fps :", fps)
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

prop = cv2.CAP_PROP_FRAME_COUNT if not imutils.is_cv2() else cv2.cv.CV_CAP_PROP_FRAME_COUNT
total = int(video.get(prop)) if video.get(prop) else 0

# ------------------- Load models -------------------
width, height = 640, 360  # TrackNet expects 640x360
m = trackNet(args.n_classes, input_height=height, input_width=width)
m.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
m.load_weights(args.save_weights_path)

LABELS = open(args.path_yolo_classes).read().strip().split("\n")
net = cv2.dnn.readNet(args.path_yolo_weights, args.path_yolo_config)

# ------------------- Resume logic -------------------
sig = video_signature(input_video_path)
start_frame = 0
append_header = True  # whether CSV needs a header

if args.resume:
    ck = load_ckpt(ckpt_path)
    if ck and ck.get("video") == sig and not ck.get("complete", False):
        start_frame = int(ck.get("last_frame", -1)) + 1
        print(f"[resume] Found checkpoint at frame {start_frame-1}. Resuming from frame {start_frame}.")
        # If CSV exists, we're appending without header
        append_header = not os.path.exists(csv_path)
    else:
        # Either no ckpt, different video, or run already complete
        if ck and ck.get("complete", False):
            print("[resume] Checkpoint indicates the run completed. Starting from frame 0 (fresh CSV append).")
        else:
            print("[resume] No valid checkpoint. Starting from frame 0.")
        append_header = not os.path.exists(csv_path)

else:
    # Fresh run; if CSV exists, we will append (no header)
    append_header = not os.path.exists(csv_path)

# ------------------- CSV streaming setup -------------------
fieldnames = [
    "frame",
    "frame_path",
    "player_0_x1","player_0_y1","player_0_x2","player_0_y2",
    "player_1_x1","player_1_y1","player_1_x2","player_1_y2",
    "ball_x","ball_y"
]
csv_file = open(csv_path, "a", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
if append_header:
    csv_writer.writeheader()
    csv_file.flush()
    os.fsync(csv_file.fileno())

# ------------------- Processing loop -------------------
currentFrame = start_frame
commit_every = max(1, int(args.commit_every))

while True:
    if total > 0:
        pct = round((currentFrame / total) * 100, 2)
        print(f"Processed: {pct}% ({currentFrame}/{total})")
    else:
        print(f"Processing frame {currentFrame}...")

    # Seek + read
    video.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
    ret, img = video.read()
    if not ret:
        break

    clean_frame = img.copy()

    # -------- Player detection (YOLO) --------
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    detected_players = predict_players(outs, LABELS, img, 0.8)  # list of (x, y, w, h)

    detected_players = sorted(detected_players, key=lambda b: b[1])  # top (small y) first
    p0 = detected_players[0] if len(detected_players) > 0 else None
    p1 = detected_players[1] if len(detected_players) > 1 else None

    # -------- Ball detection (TrackNet) --------
    tn_input = cv2.resize(img, (width, height)).astype(np.float32)
    X = np.rollaxis(tn_input, 2, 0)
    pr = m.predict(np.array([X]), verbose=0)[0]
    pr = pr.reshape((height, width, args.n_classes)).argmax(axis=2).astype(np.uint8)
    heatmap = cv2.resize(pr, (output_width, output_height))
    _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7
    )

    ball_x, ball_y = "", ""
    if circles is not None and len(circles) == 1:
        ball_x = int(circles[0][0][0])
        ball_y = int(circles[0][0][1])

    # -------- Save frame (UNANNOTATED) --------
    frame_filename = os.path.join(frames_dir, f"frame_{currentFrame:06d}.jpg")
    # if resuming and the exact frame image already exists, we can skip writing it;
    # but it's cheap to overwrite and avoids partials — we'll overwrite:
    cv2.imwrite(frame_filename, clean_frame)

    # -------- CSV row --------
    row = {
        "frame": currentFrame,
        "frame_path": frame_filename,
        "player_0_x1": "", "player_0_y1": "", "player_0_x2": "", "player_0_y2": "",
        "player_1_x1": "", "player_1_y1": "", "player_1_x2": "", "player_1_y2": "",
        "ball_x": ball_x, "ball_y": ball_y,
    }
    if p0 is not None:
        x, y, w, h = p0
        row["player_0_x1"], row["player_0_y1"], row["player_0_x2"], row["player_0_y2"] = x, y, x+w, y+h
    if p1 is not None:
        x, y, w, h = p1
        row["player_1_x1"], row["player_1_y1"], row["player_1_x2"], row["player_1_y2"] = x, y, x+w, y+h

    csv_writer.writerow(row)

    # periodic durability + checkpoint
    if (currentFrame - start_frame + 1) % commit_every == 0:
        csv_file.flush()
        os.fsync(csv_file.fileno())
        save_ckpt(ckpt_path, sig, currentFrame, total, complete=False)

    currentFrame += 1

# ------------------- finalize -------------------
video.release()
csv_file.flush()
os.fsync(csv_file.fileno())
csv_file.close()
save_ckpt(ckpt_path, sig, currentFrame - 1, total, complete=True)
print(f"CSV saved to: {csv_path}")
print(f"Frames saved to: {frames_dir}")
print(f"Checkpoint saved to: {ckpt_path} (complete=True)")
