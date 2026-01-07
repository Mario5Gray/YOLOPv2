#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import cv2
import numpy as np
from rknnlite.api import RKNNLite


class YOLOPv2RKNN:
    def __init__(self, rknn_path, confThreshold=0.5):
        self.confThreshold = float(confThreshold)
        self.nmsThreshold = 0.5

        # COCO 80 classes assumed by head=85 and repo logs (data/coco.yaml)
        # This file is OPTIONAL: only for display names.
        try:
            self.classes = [x.strip() for x in open("coco.names", "r").readlines() if x.strip()]
        except Exception:
            self.classes = [str(i) for i in range(80)]
        self.num_class = len(self.classes)

        # Input config (matches your model: NCHW 640x640)
        self.input_height = 640
        self.input_width = 640

        # Anchors from the ONNX demo you pasted (IMPORTANT)
        self.anchors = [
             [(12,16), (19,36), (40,28)],        # stride 8  -> 80x80
             [(36,75), (76,55), (72,146)],       # stride 16 -> 40x40
             [(142,110), (192,243), (459,401)],  # stride 32 -> 20x20
        ]
        self.na = len(self.anchors[0]) // 2
        self.no = self.num_class + 5
        self.stride = [8, 16, 32]
        self.nl = len(self.stride)
        self.anchors = [np.array(a, np.float32).reshape(1,3,1,1,2) for a in self.anchors]

        # Precompute grids at 640 for each stride
        # This runs once in __init__
        self.grid = [self._make_grid(nx, ny) for nx, ny in [(80, 80), (40, 40), (20, 20)]]
        for i in range(self.nl):
            h = int(self.input_height / self.stride[i])
            w = int(self.input_width / self.stride[i])
            self.grid.append(self._make_grid(w, h))

        # RKNN runtime
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def release(self):
        try:
            self.rknn.release()
        except Exception:
            pass

    def _make_grid(self, nx, ny):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return np.stack((xv, yv), 2).reshape(1,1,ny,nx,2).astype(np.float32)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        label = f"{self.classes[classId]}:{conf:.2f}"
        cv2.putText(frame, label, (left, max(0, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
        return frame

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def detect(self, frame_bgr):
        cv2.imwrite("DEBUG_input_frame.png", frame_bgr)
        print("DEBUG frame shape:", frame_bgr.shape, "dtype:", frame_bgr.dtype)
        image_width, image_height = frame_bgr.shape[1], frame_bgr.shape[0]
        ratioh = image_height / self.input_height
        ratiow = image_width / self.input_width

        # --- Preprocess (match your RKNN build: feed uint8 0..255) ---
        input_image = cv2.resize(frame_bgr, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.transpose(2, 0, 1)                # CHW
        input_image = np.expand_dims(input_image, axis=0).astype(np.uint8)  # NCHW uint8

        # --- Inference ---
        # Expected outs:
        # results[0] drivable (1,2,640,640)
        # results[1] lane     (1,1,640,640)
        # results[2:5] det heads (1,255,80/40/20,80/40/20)
        results = self.rknn.inference(inputs=[input_image], data_format=['nchw'])
        assert results[2].shape[2:] == (80,80)
        assert results[3].shape[2:] == (40,40)
        assert results[4].shape[2:] == (20,20)        
        for k, o in enumerate(results):
            a = np.array(o)
            print(k, a.shape, a.dtype)

        # --- Detection decode (ported from your ONNX demo) ---
        z = []
        for i in range(3):
            head = np.array(results[i + 2])          # 2,3,4 are det heads
            bs, _, ny, nx = head.shape

            y = head.reshape(bs, 3, 5 + self.num_class, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self._sigmoid(y)
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * self.anchors[i]

            z.append(y.reshape(bs, -1, 5 + self.num_class))

        det_out = np.concatenate(z, axis=1).squeeze(axis=0)
        
        boxes, confidences, classIds = [], [], []
        for i in range(det_out.shape[0]):
            obj = det_out[i, 4]
            cls_scores = det_out[i, 5:]
            cls_id = np.argmax(cls_scores)
            conf = obj * cls_scores[cls_id]

            if conf < self.confThreshold:
                continue

            cx, cy, w, h = det_out[i, :4]
            x = int((cx - 0.5 * w) * ratiow)
            y = int((cy - 0.5 * h) * ratioh)
            width = int(w * ratiow)
            height = int(h * ratioh)

            boxes.append([x, y, width, height])
            classIds.append(cls_id)
            confidences.append(conf)

            
        # OpenCV NMSBoxes returns indices in different shapes depending on version
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                left, top, width, height = boxes[i]
                self.drawPred(frame_bgr, classIds[i], confidences[i],
                              left, top, left + width, top + height)

        # --- Drivable Area Segmentation (same as ONNX demo) ---
        drivable_area = np.squeeze(np.array(results[0]), axis=0)  # (2,640,640)
        mask = np.argmax(drivable_area, axis=0).astype(np.uint8)
        mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame_bgr[mask == 1] = [0, 255, 0]
        
        # --- Lane Line (same as ONNX demo) ---
        lane_line = np.squeeze(np.array(results[1]))  # (640,640)
        lane_mask = np.where(lane_line > 0.5, 1, 0).astype(np.uint8)
        lane_mask = cv2.resize(lane_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        frame_bgr[lane_mask == 1] = [255, 0, 0]


        return frame_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", required=True, help="path to .rknn")
    parser.add_argument("--img", required=True, help="path to image")
    parser.add_argument("--conf", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--out", default="out_rknn.png", help="output image")
    args = parser.parse_args()

    net = YOLOPv2RKNN(args.rknn, confThreshold=args.conf)
    try:
        src = cv2.imread(args.img)
        if src is None:
            raise RuntimeError(f"failed to read image: {args.img}")
        out = net.detect(src)
        cv2.imwrite(args.out, out)
        print("Wrote:", args.out)
    finally:
        net.release()


if __name__ == "__main__":
    main()
