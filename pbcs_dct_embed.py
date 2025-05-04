import cv2
import numpy as np
import os
import random
import shutil
import subprocess
import platform
from scipy.fftpack import dct

def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(frames_dir, exist_ok=True)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(frames_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(path, frame)
        idx += 1
    cap.release()
    return idx

def extract_audio(video_path, audio_path="extracted_audio.aac"):
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path])

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def compute_complexity_dct(block):
    return np.std(dct2(block))

def embed_lsb_in_block(block, bit):
    flat = block.flatten()
    flat[0] = (flat[0] & 254) | int(bit)  # 254 = 0b11111110
    return flat.reshape(block.shape)


def embed_lsb_using_dct(gray, secret_bits, threshold=10.0, block_size=8):
    h, w = gray.shape
    secret_idx = 0
    loc_map = []
    modified = gray.copy()

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            if secret_idx >= len(secret_bits):
                break
            block = gray[i:i+block_size, j:j+block_size]
            if compute_complexity_dct(block) > threshold:
                new_block = embed_lsb_in_block(block.copy(), secret_bits[secret_idx])
                modified[i:i+block_size, j:j+block_size] = new_block
                loc_map.append((i, j))
                secret_idx += 1

    return modified, loc_map

def main():
    input_video = input("Nhập video đầu vào: ")
    message = input("Nhập thông điệp cần giấu: ")
    secret_bits = ''.join(format(ord(c), '08b') for c in message)

    frames_dir = "frames_temp"
    audio_path = "extracted_audio.aac"
    frame_count = extract_frames(input_video, frames_dir)
    fps = get_video_fps(input_video)
    extract_audio(input_video, audio_path)

    chosen_frame = random.randint(0, frame_count - 1)
    print(f"Khung hình được chọn: {chosen_frame}")
    gray = cv2.cvtColor(cv2.imread(f"{frames_dir}/frame_{chosen_frame:05d}.png"), cv2.COLOR_BGR2GRAY)

    modified, loc_map = embed_lsb_using_dct(gray, secret_bits)
    mod_color = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{frames_dir}/frame_{chosen_frame:05d}.png", mod_color)

    with open("key.txt", "w") as f:
        f.write(f"frame_index: {chosen_frame}\n")
        f.write(f"locations: {loc_map}\n")
        f.write(f"bit_length: {len(secret_bits)}\n")

    output_noaudio = "output_stego_noaudio.mp4"
    null = "NUL" if platform.system() == "Windows" else "/dev/null"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", f"{frames_dir}/frame_%05d.png",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", output_noaudio
    ])
    final_output = "output_stego.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", output_noaudio, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", final_output
    ])
    print(f"Video đã giấu tin: {final_output}")

if __name__ == "__main__":
    main()
