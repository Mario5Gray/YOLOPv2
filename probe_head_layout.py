#!/usr/bin/env python3
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite

def sigmoid(x): return 1/(1+np.exp(-x))

def preprocess_uint8_rgb(img_bgr):
    img = cv2.resize(img_bgr, (640,640), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.transpose(img, (2,0,1))[None, ...].astype(np.uint8)
    return x

def obj_heatmaps(head):
    """
    head: (1,255,S,S)
    Return dict name->(S,S) objectness map in [0,1] for several layouts.
    """
    o = np.array(head)[0]  # (255,S,S)
    C,S,_ = o.shape
    nc = 80
    na = 3
    no = 5 + nc
    assert C == na*no, (C, na, no)

    maps = {}

    # Layout A (what we assumed): anchor-major blocks: [a0(no), a1(no), a2(no)]
    A = o.reshape(na, no, S, S)
    maps["A_anchor_major"] = sigmoid(A[:,4,:,:]).max(axis=0)

    # Layout B: attribute-major: [tx(all anchors), ty(all), tw(all), ... cls...]
    B = o.reshape(no, na, S, S)
    maps["B_attr_major"] = sigmoid(B[4,:,:,:]).max(axis=0)

    # Layout C: treat as (S,S,255) then reshape
    C0 = o.transpose(1,2,0)  # (S,S,255)
    C1 = C0.reshape(S, S, na, no)
    maps["C_S_S_na_no"] = sigmoid(C1[:,:, :, 4]).max(axis=2)

    # Layout D: (S,S,no,na)
    D1 = C0.reshape(S, S, no, na)
    maps["D_S_S_no_na"] = sigmoid(D1[:,:, 4, :]).max(axis=2)

    return maps

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} model.rknn image.jpg out_prefix")
        return 2

    model_path, image_path, out_prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Failed to read:", image_path)
        return 2

    rknn = RKNNLite()
    assert rknn.load_rknn(model_path) == 0
    assert rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0
    outs = rknn.inference(inputs=[preprocess_uint8_rgb(img_bgr)])
    rknn.release()

    # Use only the 80x80 head for this probe (outs[2])
    head = outs[2]
    maps = obj_heatmaps(head)

    for name, m in maps.items():
        u8 = np.clip(m * 255, 0, 255).astype(np.uint8)
        u8 = cv2.resize(u8, (640,640), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{out_prefix}_{name}.png", u8)
        print("wrote", f"{out_prefix}_{name}.png", "min/max", float(m.min()), float(m.max()))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
