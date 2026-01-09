import argparse
import time
from collections import deque
from vidgear.gears import CamGear, WriteGear
from utils import YOLOPv2RKNN
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", required=True, help="path to .rknn")
    parser.add_argument("--vid", required=True, help="path to video")
    parser.add_argument("--conf", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--out", default="out_rknn.mp4", help="output video")
    parser.add_argument("--br", default="5M", help="output bitrate")
    parser.add_argument("--ffmpeg", default="/opt/ffmpeg-rockchip/bin/ffmpeg", help="Path to ffmpeg")
    args = parser.parse_args()

    net = YOLOPv2RKNN(args.rknn, confThreshold=args.conf)

    # Input stream - CamGear doesn't support custom ffmpeg params directly
    # It uses OpenCV backend, so hardware accel needs to be via OpenCV build
    stream = CamGear(
        source=args.vid,
        logging=True
    ).start()

    # Get first frame
    frame = stream.read()  # NOT: ret, frame = stream.read()
    
    if frame is None:
        print("Failed to read video")
        return

    height, width = frame.shape[:2]
    fps = stream.framerate if hasattr(stream, 'framerate') else 30.0
    bitrate = args.br

    # Output with hardware encoding
    output_params = {
        "-b:v": bitrate,
        "-input_framerate": fps,
        "-vcodec": "hevc_rkmpp" 
    }

    writer = WriteGear(
        custom_ffmpeg=args.ffmpeg,
        output=args.out,  # Changed from 'output' to 'output_filename'
        logging=True,
        **output_params
    )

    # Process and write first frame
    out_frame = net.detect(frame)
    writer.write(out_frame)

    # --- timing helpers ---
    infer_ms_window = deque(maxlen=30)
    loop_ms_window  = deque(maxlen=30)

    infer_ms_total = 0.0
    loop_ms_total  = 0.0
    n_frames = 0

    # If you want higher-resolution timing on Linux:
    now = time.perf_counter  # monotonic, high-res

    # Loop over frames
    while True:
        frame = stream.read()  # VidGear returns just frame, not (ret, frame)
        
        if frame is None:  # Changed from 'if not frame'
            break

        t_loop0 = now()
        start = time.time()
        out_frame = net.detect(frame)
        inference_fps = 1.0 / (time.time() - start)

        #Optional FPS overlay
        cv2.putText(out_frame, f"FPS: {inference_fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # --- inference timing ---
        t0 = now()
        out_frame = net.detect(frame)
        t1 = now()

        infer_ms = (t1 - t0) * 1000.0
        infer_ms_window.append(infer_ms)
        infer_ms_total += infer_ms

        # --- (optional) total loop timing (includes overlay+write) ---
        # Do overlay/write first, then measure loop end
        infer_fps = 1000.0 / max(1e-6, infer_ms)

        # rolling stats
        infer_ms_avg = float(np.mean(infer_ms_window))
        infer_fps_avg = 1000.0 / max(1e-6, infer_ms_avg)

        writer.write(out_frame)

        t_loop1 = now()
        loop_ms = (t_loop1 - t_loop0) * 1000.0
        loop_ms_window.append(loop_ms)
        loop_ms_total += loop_ms

        n_frames += 1

        # Optional: print every N frames
        if n_frames % 5 == 0:
            loop_ms_avg = float(np.mean(loop_ms_window))
            print(f"[{n_frames}] infer={infer_ms_avg:.2f}ms avg30, loop={loop_ms_avg:.2f}ms avg30")

    # Cleanup
    stream.stop()
    writer.close()

    # --- summary ---
    if n_frames > 0:
        print("\n=== Timing summary ===")
        print(f"Frames: {n_frames}")
        print(f"Avg infer: {infer_ms_total / n_frames:.2f} ms  ({1000.0 / max(1e-6, infer_ms_total / n_frames):.2f} FPS)")
        print(f"Avg loop : {loop_ms_total  / n_frames:.2f} ms  ({1000.0 / max(1e-6, loop_ms_total  / n_frames):.2f} FPS)")

if __name__ == "__main__":
    main()
