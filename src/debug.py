from pathlib import Path
import zipfile

src_dir = Path(r"D:\Development\Projects\SignAssistant\src")
task_path = src_dir / "pose_landmarker.task"
out_path = src_dir / "pose_landmarker.tflite"

with zipfile.ZipFile(task_path, "r") as archive:
    print("Files in archive:", archive.namelist())
    # Explicitly extract the landmarks detector, NOT the bounding box detector!
    model_bytes = archive.read("pose_landmarks_detector.tflite")
    with open(out_path, "wb") as f:
        f.write(model_bytes)

print(f"[✓] Extracted pose_landmarks_detector.tflite -> {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")