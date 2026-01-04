import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger(__name__)


@dataclass
class MoveNetConfig:
    model_path: str
    device: str = "cpu"  # "cpu" or "cuda"/"cuda:0" if CUDAExecutionProvider available
    input_size: int | None = None  # if None, inferred from ONNX graph


class MoveNetONNX:
    """Minimal MoveNet SinglePose (Lightning) ONNX wrapper.

    Expects a SinglePose model exported to ONNX (e.g. 192x192, NHWC) with output shape
    [1, 1, 17, 3] or [1, 17, 3] in (y, x, confidence) normalized coordinates.
    """

    def __init__(self, cfg: MoveNetConfig) -> None:
        self.cfg = cfg
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"MoveNet ONNX model not found: {model_path}")

        providers: List[str] = ["CPUExecutionProvider"]
        if cfg.device.lower().startswith("cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                LOGGER.warning(
                    "CUDAExecutionProvider is not available in onnxruntime. "
                    "Falling back to CPUExecutionProvider.",
                )

        LOGGER.info("Loading MoveNet ONNX from %s with providers=%s", model_path, providers)
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_type = inp.type  # e.g. 'tensor(float)' or 'tensor(int32)'
        input_shape = inp.shape
        # Typical MoveNet: [1, 192, 192, 3] (NHWC)
        self.is_nhwc = input_shape[-1] == 3
        if self.is_nhwc:
            self.input_height = int(input_shape[1])
            self.input_width = int(input_shape[2])
        else:
            # NCHW
            self.input_height = int(input_shape[2])
            self.input_width = int(input_shape[3])

        if cfg.input_size is not None:
            self.input_height = cfg.input_size
            self.input_width = cfg.input_size

    def _preprocess(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> tuple[np.ndarray, Tuple[int, int, int, int]] | None:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            crop_rgb,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Prepare input tensor depending on ONNX model's expected dtype.
        if self.input_type == "tensor(int32)":
            # Quantized MoveNet variants expect integer pixel values in [0, 255].
            inp = resized.astype(np.int32)
        else:
            # Default: float32 in [0, 1].
            inp = resized.astype(np.float32) / 255.0

        if self.is_nhwc:
            inp = inp[None, ...]  # [1, H, W, 3]
        else:
            inp = np.transpose(inp, (2, 0, 1))[None, ...]  # [1, 3, H, W]

        return inp, (x1, y1, x2, y2)

    def infer(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Run MoveNet on a cropped person ROI.

        Returns
        -------
        np.ndarray
            Array of shape [num_keypoints, 3] with (x, y, confidence) in full-frame
            pixel coordinates. If inference fails, returns an empty array.
        """

        pre = self._preprocess(frame_bgr, bbox_xyxy)
        if pre is None:
            return np.zeros((0, 3), dtype=np.float32)

        inp, (x1, y1, x2, y2) = pre
        outputs = self.session.run(None, {self.input_name: inp})

        kpts = np.array(outputs[0], dtype=np.float32)
        kpts = kpts.reshape(-1, 3)  # [num_keypoints, 3]

        # MoveNet uses (y, x, confidence) in normalized [0, 1] coords of the crop.
        ys = kpts[:, 0]
        xs = kpts[:, 1]
        scores = kpts[:, 2]

        w = float(x2 - x1)
        h = float(y2 - y1)

        xs_img = x1 + xs * w
        ys_img = y1 + ys * h

        return np.stack([xs_img, ys_img, scores], axis=1)
