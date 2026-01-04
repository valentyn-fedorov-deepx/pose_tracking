import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from .movenet_onnx import MoveNetConfig, MoveNetONNX


LOGGER = logging.getLogger("pose_tracker")


@dataclass
class PoseTrackingConfig:
    model_path: str  # YOLO model for pose or detection depending on backend
    device: str
    tracker: str
    conf: float
    iou: float
    data_format: str  # "json" or "csv"
    backend: str  # "yolo_pose" or "movenet_yolo_roi"
    detector_model: Optional[str] = None  # for MoveNet pipeline (YOLO detect)
    movenet_model: Optional[str] = None  # path to MoveNet ONNX


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 Pose Tracking - Server-ready batch processor",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="input",
        help="Path to input video file or directory (default: input)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="output",
        help="Directory to store processed videos and analytics (default: output)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n-pose.pt",
        help="YOLOv8 pose model path or name (e.g. yolov8n-pose.pt, yolov8m-pose.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device, e.g. 'cpu', 'cuda', 'cuda:0' (default: cpu)",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration to use (default: bytetrack.yaml)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )

    parser.add_argument(
        "--data-format",
        choices=["json", "csv"],
        default="json",
        help="Output format for per-frame skeleton analytics (default: json)",
    )

    parser.add_argument(
        "--backend",
        choices=["yolo_pose", "movenet_yolo_roi"],
        default="yolo_pose",
        help="Pose backend: YOLO pose head or YOLO detect + MoveNet (default: yolo_pose)",
    )

    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO detection model for MoveNet pipeline (default: yolov8n.pt)",
    )

    parser.add_argument(
        "--movenet-model",
        type=str,
        default="models/movenet_singlepose_lightning_192x192.onnx",
        help="MoveNet SinglePose Lightning ONNX path for MoveNet pipeline",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Run both backends (yolo_pose and movenet_yolo_roi) on the same videos "
            "and save a benchmark_summary.json with runtime metrics."
        ),
    )

    return parser.parse_args()


def collect_video_paths(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {input_path}")

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)

    if not videos:
        raise FileNotFoundError(f"No video files found under directory: {input_path}")

    return videos


def ensure_output_dir(output_dir: str) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        LOGGER.warning("Failed to open video for FPS read: %s", video_path)
        return 30.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 0:
        LOGGER.warning("Invalid FPS reported for %s, defaulting to 30.0", video_path)
        return 30.0

    return float(fps)


def extract_frame_skeletons(
    result: Any,
    frame_index: int,
    fps: float,
    video_name: str,
    track_history: Dict[int, Dict[str, float]],
    track_paths: Dict[int, List[tuple]],
    max_path_length: int,
) -> Dict[str, Any]:
    frame_time = frame_index / fps if fps > 0 else None
    frame_record: Dict[str, Any] = {
        "video": video_name,
        "frame_index": frame_index,
        "timestamp": frame_time,
        "detections": [],
    }

    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if keypoints is None or keypoints.data is None:
        return frame_record

    kpt_tensor = keypoints.data.cpu().numpy()  # shape: [num_instances, num_keypoints, 3]
    num_instances = kpt_tensor.shape[0]

    track_ids: Optional[List[Optional[int]]] = None
    if boxes is not None and getattr(boxes, "id", None) is not None:
        # boxes.id is a tensor of shape [num_instances]
        track_ids = boxes.id.int().cpu().tolist()

    centers: Optional[List[Optional[tuple]]] = None
    if boxes is not None and getattr(boxes, "xywh", None) is not None:
        xywh = boxes.xywh.cpu().numpy()
        centers = [(float(b[0]), float(b[1])) for b in xywh]

    for i in range(num_instances):
        instance_keypoints = kpt_tensor[i]  # [num_keypoints, 3]
        keypoints_list: List[Dict[str, Any]] = []

        for kp in instance_keypoints:
            # Expected layout: [x, y, confidence]
            x, y = float(kp[0]), float(kp[1])
            conf = float(kp[2]) if kp.shape[0] > 2 else None
            keypoints_list.append(
                {
                    "x": x,
                    "y": y,
                    "confidence": conf,
                }
            )

        track_id: Optional[int] = None
        if track_ids is not None and i < len(track_ids):
            tid = track_ids[i]
            track_id = int(tid) if tid is not None else None

        motion = None
        if track_id is not None and centers is not None and i < len(centers):
            cx, cy = centers[i]
            prev = track_history.get(track_id)
            if prev is not None:
                dx = cx - prev["cx"]
                dy = cy - prev["cy"]
            else:
                dx = 0.0
                dy = 0.0
            track_history[track_id] = {"cx": cx, "cy": cy}
            motion = {"dx": dx, "dy": dy, "cx": cx, "cy": cy}

            # Update path history for visualization (trajectory polyline)
            path = track_paths.setdefault(track_id, [])
            path.append((cx, cy))
            if len(path) > max_path_length:
                path.pop(0)

        frame_record["detections"].append(
            {
                "track_id": track_id,
                "motion": motion,
                "keypoints": keypoints_list,
            }
        )

    return frame_record


def write_frame_skeletons_to_csv(
    csv_writer: Any,
    frame_record: Dict[str, Any],
) -> None:
    video_name = frame_record["video"]
    frame_index = frame_record["frame_index"]
    timestamp = frame_record["timestamp"]

    for det in frame_record["detections"]:
        track_id = det["track_id"]
        motion = det.get("motion") if isinstance(det, dict) else None
        dx = motion.get("dx") if motion is not None else None
        dy = motion.get("dy") if motion is not None else None
        for k_idx, kp in enumerate(det["keypoints"]):
            csv_writer.writerow(
                [
                    video_name,
                    frame_index,
                    timestamp,
                    track_id,
                    k_idx,
                    kp["x"],
                    kp["y"],
                    kp["confidence"],
                    dx,
                    dy,
                ]
            )


def process_video_yolo_pose(
    model: YOLO,
    video_path: Path,
    output_dir: Path,
    cfg: PoseTrackingConfig,
) -> Dict[str, Any]:
    LOGGER.info("Processing video: %s", video_path)

    fps = get_video_fps(video_path)
    video_name = video_path.stem

    # Get total frame count for progress bar (if available)
    total_frames: Optional[int] = None
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            total_frames = total
    cap.release()

    output_video_path = output_dir / f"{video_name}_pose_tracked.mp4"

    if cfg.data_format == "json":
        output_data_path = output_dir / f"{video_name}_pose_data.json"
    else:
        output_data_path = output_dir / f"{video_name}_pose_data.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer: Optional[cv2.VideoWriter] = None

    frames_data: List[Dict[str, Any]] = []
    csv_file = None
    csv_writer = None
    # Track history for motion vectors (per track ID)
    track_history: Dict[int, Dict[str, float]] = {}
    # Path history for drawing trajectories (per track ID)
    track_paths: Dict[int, List[tuple]] = {}
    max_path_length = 30

    if cfg.data_format == "csv":
        import csv

        csv_file = open(output_data_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "video",
                "frame_index",
                "timestamp",
                "track_id",
                "keypoint_index",
                "x",
                "y",
                "confidence",
                "dx",
                "dy",
            ]
        )

    pbar = tqdm(
        total=total_frames,
        desc=f"{video_name} [yolo_pose]",
        unit="frame",
    )

    t_start = time.perf_counter()

    try:
        results_generator = model.track(
            source=str(video_path),
            stream=True,
            tracker=cfg.tracker,
            persist=True,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            verbose=False,
        )

        frame_index = 0

        for result in results_generator:
            # First, update analytics and track histories (including paths)
            frame_record = extract_frame_skeletons(
                result=result,
                frame_index=frame_index,
                fps=fps,
                video_name=video_path.name,
                track_history=track_history,
                track_paths=track_paths,
                max_path_length=max_path_length,
            )

            # Get annotated frame from YOLO (boxes + pose)
            annotated_frame = result.plot()

            # Draw trajectory polylines for each active track on top of YOLO annotations
            for _, points in track_paths.items():
                if len(points) < 2:
                    continue
                pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    annotated_frame,
                    [pts],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=3,
                )

            if video_writer is None:
                h, w = annotated_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    fps,
                    (w, h),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")

            video_writer.write(annotated_frame)

            if cfg.data_format == "json":
                frames_data.append(frame_record)
            else:
                assert csv_writer is not None
                write_frame_skeletons_to_csv(csv_writer, frame_record)

            frame_index += 1
            pbar.update(1)

    finally:
        elapsed = time.perf_counter() - t_start
        pbar.close()

        if video_writer is not None:
            video_writer.release()

        if csv_file is not None:
            csv_file.close()

    if cfg.data_format == "json":
        payload = {
            "video": video_path.name,
            "backend": "yolo_pose",
            "model": cfg.model_path,
            "tracker": cfg.tracker,
            "fps": fps,
            "frames": frames_data,
        }
        with open(output_data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    LOGGER.info("Saved processed video to: %s", output_video_path)
    LOGGER.info("Saved pose analytics to: %s", output_data_path)

    metrics = {
        "video": video_path.name,
        "backend": "yolo_pose",
        "pose_model": cfg.model_path,
        "detector_model": None,
        "frame_count": frame_index,
        "total_time_sec": elapsed,
        "avg_fps": (frame_index / elapsed) if elapsed > 0 and frame_index > 0 else None,
        "avg_latency_ms": (elapsed / frame_index * 1000.0)
        if frame_index > 0 and elapsed > 0
        else None,
    }

    return metrics


def process_video_movenet_yolo_roi(
    det_model: YOLO,
    movenet: MoveNetONNX,
    video_path: Path,
    output_dir: Path,
    cfg: PoseTrackingConfig,
) -> Dict[str, Any]:
    """YOLO detect + ByteTrack for IDs, MoveNet for pose keypoints."""
    LOGGER.info("Processing video (MoveNet pipeline): %s", video_path)

    fps = get_video_fps(video_path)
    video_name = video_path.stem

    # Get total frame count for progress bar (if available)
    total_frames: Optional[int] = None
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            total_frames = total
    cap.release()

    output_video_path = output_dir / f"{video_name}_pose_tracked_movenet.mp4"

    if cfg.data_format == "json":
        output_data_path = output_dir / f"{video_name}_pose_data_movenet.json"
    else:
        output_data_path = output_dir / f"{video_name}_pose_data_movenet.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer: Optional[cv2.VideoWriter] = None

    frames_data: List[Dict[str, Any]] = []
    csv_file = None
    csv_writer = None
    track_history: Dict[int, Dict[str, float]] = {}
    track_paths: Dict[int, List[tuple]] = {}
    max_path_length = 30

    if cfg.data_format == "csv":
        import csv

        csv_file = open(output_data_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "video",
                "frame_index",
                "timestamp",
                "track_id",
                "keypoint_index",
                "x",
                "y",
                "confidence",
                "dx",
                "dy",
            ]
        )

    pbar = tqdm(
        total=total_frames,
        desc=f"{video_name} [movenet_yolo_roi]",
        unit="frame",
    )

    t_start = time.perf_counter()

    try:
        results_generator = det_model.track(
            source=str(video_path),
            stream=True,
            tracker=cfg.tracker,
            persist=True,
            conf=cfg.conf,
            iou=cfg.iou,
            device=cfg.device,
            verbose=False,
        )

        frame_index = 0

        for result in results_generator:
            frame = result.orig_img  # BGR frame
            boxes = getattr(result, "boxes", None)

            if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
                # Nothing detected: still write frame and continue
                annotated_frame = frame.copy()
            else:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids: Optional[List[Optional[int]]] = None
                if getattr(boxes, "id", None) is not None:
                    track_ids = boxes.id.int().cpu().tolist()

                detections = []
                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = xyxy[i]
                    bbox = (float(x1), float(y1), float(x2), float(y2))
                    kpts = movenet.infer(frame, bbox)
                    if kpts.shape[0] == 0:
                        continue

                    keypoints_list: List[Dict[str, Any]] = []
                    for kp in kpts:
                        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                        keypoints_list.append({"x": x, "y": y, "confidence": conf})

                    track_id: Optional[int] = None
                    if track_ids is not None and i < len(track_ids):
                        tid = track_ids[i]
                        track_id = int(tid) if tid is not None else None

                    # Motion + path using bbox center (for consistency with YOLO pose)
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    motion = None
                    if track_id is not None:
                        prev = track_history.get(track_id)
                        if prev is not None:
                            dx = cx - prev["cx"]
                            dy = cy - prev["cy"]
                        else:
                            dx = 0.0
                            dy = 0.0
                        track_history[track_id] = {"cx": cx, "cy": cy}
                        motion = {"dx": dx, "dy": dy, "cx": cx, "cy": cy}

                        path = track_paths.setdefault(track_id, [])
                        path.append((cx, cy))
                        if len(path) > max_path_length:
                            path.pop(0)

                    detections.append(
                        {
                            "track_id": track_id,
                            "motion": motion,
                            "keypoints": keypoints_list,
                            "bbox": bbox,
                        }
                    )

                # Draw boxes + skeletons + trajectories
                annotated_frame = frame.copy()
                for det in detections:
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    tid = det["track_id"]
                    if tid is not None:
                        cv2.putText(
                            annotated_frame,
                            f"ID {tid}",
                            (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    # Simple COCO skeleton connections (indices depend on MoveNet topology)
                    kps = det["keypoints"]
                    for kp in kps:
                        cv2.circle(
                            annotated_frame,
                            (int(kp["x"]), int(kp["y"])),
                            3,
                            (0, 255, 0),
                            -1,
                        )

                # Trajectory polylines
                for _, points in track_paths.items():
                    if len(points) < 2:
                        continue
                    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(
                        annotated_frame,
                        [pts],
                        isClosed=False,
                        color=(0, 0, 255),
                        thickness=3,
                    )

                # Build frame_record compatible with analytics schema
                frame_time = frame_index / fps if fps > 0 else None
                frame_record = {
                    "video": video_path.name,
                    "frame_index": frame_index,
                    "timestamp": frame_time,
                    "detections": [
                        {
                            "track_id": det["track_id"],
                            "motion": det["motion"],
                            "keypoints": det["keypoints"],
                        }
                        for det in detections
                    ],
                }

                if cfg.data_format == "json":
                    frames_data.append(frame_record)
                else:
                    assert csv_writer is not None
                    write_frame_skeletons_to_csv(csv_writer, frame_record)

            if video_writer is None:
                h, w = annotated_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output_video_path),
                    fourcc,
                    fps,
                    (w, h),
                )

                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to open VideoWriter for {output_video_path}")

            video_writer.write(annotated_frame)

            frame_index += 1
            pbar.update(1)

    finally:
        elapsed = time.perf_counter() - t_start
        pbar.close()

        if video_writer is not None:
            video_writer.release()

        if csv_file is not None:
            csv_file.close()

    if cfg.data_format == "json":
        payload = {
            "video": video_path.name,
            "backend": "movenet_yolo_roi",
            "model": cfg.movenet_model,
            "detector_model": cfg.detector_model,
            "tracker": cfg.tracker,
            "fps": fps,
            "frames": frames_data,
        }
        with open(output_data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    LOGGER.info("Saved processed video to: %s", output_video_path)
    LOGGER.info("Saved pose analytics to: %s", output_data_path)

    metrics = {
        "video": video_path.name,
        "backend": "movenet_yolo_roi",
        "pose_model": cfg.movenet_model,
        "detector_model": cfg.detector_model,
        "frame_count": frame_index,
        "total_time_sec": elapsed,
        "avg_fps": (frame_index / elapsed) if elapsed > 0 and frame_index > 0 else None,
        "avg_latency_ms": (elapsed / frame_index * 1000.0)
        if frame_index > 0 and elapsed > 0
        else None,
    }

    return metrics


def main() -> None:
    setup_logging()
    args = parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    cfg = PoseTrackingConfig(
        model_path=args.model,
        device=args.device,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        data_format=args.data_format,
        backend=args.backend,
        detector_model=args.detector_model,
        movenet_model=args.movenet_model,
    )

    video_paths = collect_video_paths(args.input)
    LOGGER.info("Found %d video(s) to process", len(video_paths))

    metrics_all: List[Dict[str, Any]] = []

    if args.compare:
        LOGGER.info(
            "Running comparison mode: YOLO pose backend vs YOLO detect + MoveNet backend.",
        )

        LOGGER.info("Loading YOLO pose model: %s", cfg.model_path)
        yolo_pose_model = YOLO(cfg.model_path)

        LOGGER.info("Loading YOLO detector model for MoveNet pipeline: %s", cfg.detector_model)
        det_model = YOLO(cfg.detector_model)

        movenet_cfg = MoveNetConfig(
            model_path=cfg.movenet_model,
            device=cfg.device,
        )
        movenet = MoveNetONNX(movenet_cfg)

        for vp in video_paths:
            try:
                m1 = process_video_yolo_pose(
                    model=yolo_pose_model,
                    video_path=vp,
                    output_dir=output_dir,
                    cfg=cfg,
                )
                m2 = process_video_movenet_yolo_roi(
                    det_model=det_model,
                    movenet=movenet,
                    video_path=vp,
                    output_dir=output_dir,
                    cfg=cfg,
                )
                metrics_all.extend([m1, m2])
            except Exception:
                LOGGER.exception("Failed to process video in comparison mode: %s", vp)

    else:
        if cfg.backend == "yolo_pose":
            LOGGER.info("Loading YOLO pose model: %s", cfg.model_path)
            model = YOLO(cfg.model_path)

            for vp in video_paths:
                try:
                    m = process_video_yolo_pose(
                        model=model,
                        video_path=vp,
                        output_dir=output_dir,
                        cfg=cfg,
                    )
                    metrics_all.append(m)
                except Exception:
                    LOGGER.exception("Failed to process video: %s", vp)
        else:
            LOGGER.info(
                "Loading YOLO detector model for MoveNet pipeline: %s", cfg.detector_model,
            )
            det_model = YOLO(cfg.detector_model)

            movenet_cfg = MoveNetConfig(
                model_path=cfg.movenet_model,
                device=cfg.device,
            )
            movenet = MoveNetONNX(movenet_cfg)

            for vp in video_paths:
                try:
                    m = process_video_movenet_yolo_roi(
                        det_model=det_model,
                        movenet=movenet,
                        video_path=vp,
                        output_dir=output_dir,
                        cfg=cfg,
                    )
                    metrics_all.append(m)
                except Exception:
                    LOGGER.exception("Failed to process video: %s", vp)

    # Save benchmark summary if any metrics collected
    if metrics_all:
        summary_path = output_dir / "benchmark_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"results": metrics_all}, f, indent=2)
        LOGGER.info("Saved benchmark summary to: %s", summary_path)


if __name__ == "__main__":
    main()
