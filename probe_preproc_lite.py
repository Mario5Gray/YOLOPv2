#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite

def sigmoid(x):
    return 1/(1+np.exp(-x))

def make_input(img_bgr, rgb: bool, dtype: str):
    img = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dtype == "uint8":
        x = img.astype(np.uint8)
    elif dtype == "float01":
        x = (img.astype(np.float32) / 255.0)
    else:
        raise ValueError(dtype)

    x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
    return x

def max_obj_from_head(head):
    o = np.array(head)  # (1,255,S,S)
    b, c, h, w = o.shape
    o = o.reshape(1, 3, 85, h, w)
    obj = sigmoid(o[:, :, 4, :, :])
    return float(obj.max()), float(obj.mean())

def summarize_case(rknn, img_bgr, rgb, dtype, label):
    inp = make_input(img_bgr, rgb=rgb, dtype=dtype)
    outs = rknn.inference(inputs=[inp])

    o0 = np.array(outs[0])  # (1,2,640,640)
    o1 = np.array(outs[1])  # (1,1,640,640)
    m80, a80 = max_obj_from_head(outs[2])
    m40, a40 = max_obj_from_head(outs[3])
    m20, a20 = max_obj_from_head(outs[4])

    print(f"\n=== {label} ===")
    print("out0 min/max/mean:", float(o0.min()), float(o0.max()), float(o0.mean()))
    print("out1 min/max/mean:", float(o1.min()), float(o1.max()), float(o1.mean()))
    print("obj max/mean:", f"80 {m80:.4f}/{a80:.4f}", f"40 {m40:.4f}/{a40:.4f}", f"20 {m20:.4f}/{a20:.4f}")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} model.rknn image.jpg")
        return 2

    model_path, image_path = sys.argv[1], sys.argv[2]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Failed to read:", image_path)
        return 2

    rknn = RKNNLite()
    assert rknn.load_rknn(model_path) == 0
    assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0

    summarize_case(rknn, img_bgr, rgb=True,  dtype="uint8",  label="RGB + uint8 (expected if mean/std used in build)")
    summarize_case(rknn, img_bgr, rgb=False, dtype="uint8",  label="BGR + uint8")
    summarize_case(rknn, img_bgr, rgb=True,  dtype="float01",label="RGB + float01 (only if you DID NOT set mean/std)")
    summarize_case(rknn, img_bgr, rgb=False, dtype="float01",label="BGR + float01")

    rknn.release()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
