
import cv2
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
 
# Map your drives: Read from E: (HDD), save crops to D: (SSD/HDD)
RAW_DIR = Path(r"E:\Downloads\DeepGuard-AI-main\data\raw")
FACES_DIR = Path(r"D:\deepguard_faces")
 
DETECT_WIDTH = 640      # downscale to this width just for detection (speed)
FRAME_STRIDE = 30       # sample every Nth frame
TARGET_FACES = 10       # faces to save per video
 
 
def process_single_video(args):
    video_file, category, split = args
    out_dir = FACES_DIR / split / category / video_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Skip if faces are already extracted for this video
    if len(list(out_dir.glob("*.jpg"))) >= TARGET_FACES:
        return "skipped"
 
    # CRITICAL: stop OpenCV from spawning its own internal threads.
    # With cpu_count() processes already saturating your cores, letting
    # each process ALSO multi-thread internally causes massive
    # oversubscription -> 100% CPU with very little real throughput.
    cv2.setNumThreads(1)
 
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
 
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return "failed_open"
 
    frame_count = 0
    saved = 0
 
    while saved < TARGET_FACES:
        # Only DECODE the frame we actually need. grab() advances the
        # stream without decoding, which is what was wasting most of
        # your CPU time before (decoding 30 frames to use 1).
        if frame_count % FRAME_STRIDE != 0:
            if not cap.grab():
                break
            frame_count += 1
            continue
 
        ret, frame = cap.read()  # grab + decode, only on frames we want
        if not ret:
            break
 
        h0, w0 = frame.shape[:2]
        scale = DETECT_WIDTH / w0 if w0 > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale))) if scale != 1.0 else frame
 
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
 
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            # map detection box back to full-resolution frame
            x, y, w, h = (int(v / scale) for v in (x, y, w, h))
 
            mx, my = int(w * 0.2), int(h * 0.2)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w0, x + w + mx)
            y2 = min(h0, y + h + my)
 
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                cv2.imwrite(str(out_dir / f"frame_{frame_count}.jpg"), face_crop)
                saved += 1
 
        frame_count += 1
 
    cap.release()
    return "done"
 
 
def process_videos():
    categories = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    task_list = []
 
    for category in categories:
        cat_in = RAW_DIR / category
        if not cat_in.exists():
            print(f"Skipping {category} - folder not found in {cat_in}")
            continue
 
        videos = list(cat_in.glob("*.mp4"))
        random.seed(42)
        random.shuffle(videos)
 
        train_split = int(len(videos) * 0.8)
        val_split = int(len(videos) * 0.9)
 
        for i, video_file in enumerate(videos):
            if i < train_split:
                split = "train"
            elif i < val_split:
                split = "val"
            else:
                split = "test"
            task_list.append((video_file, category, split))
 
    print(f"Total videos queued across all categories: {len(task_list)}")
    cores = cpu_count()
    print(f"Launching multi-core extraction across {cores} CPU cores...")
 
    with Pool(processes=cores) as pool:
        # imap_unordered + tqdm gives you a live progress bar and lets
        # finished videos report back immediately instead of waiting
        # for the whole batch (pool.map) to complete.
        for _ in tqdm(pool.imap_unordered(process_single_video, task_list, chunksize=4),
                      total=len(task_list)):
            pass
 
 
if __name__ == "__main__":
    print("Starting high-speed multi-core extraction...")
    process_videos()
    print("\nExtraction completed! All face crops are ready on your D: drive.")