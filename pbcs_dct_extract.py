import cv2
import numpy as np
import ast
import re

def read_key(file="key.txt"):
    with open(file, "r") as f:
        lines = f.readlines()
        frame_idx = int(re.search(r'\d+', lines[0]).group())
        locs = ast.literal_eval(lines[1].split(":", 1)[1].strip())
        bit_len = int(re.search(r'\d+', lines[2]).group())
        return frame_idx, locs, bit_len

def extract_lsb_from_block(block):
    flat = block.flatten()
    return str(flat[0] & 1)

def extract_message(gray, locs, bit_len, block_size=8):
    bits = ""
    for i, j in locs:
        block = gray[i:i+block_size, j:j+block_size]
        bits += extract_lsb_from_block(block)
        if len(bits) >= bit_len:
            break
    return bits

def bits_to_text(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def main():
    frame_idx, locs, bit_len = read_key()
    frame = cv2.imread(f"frames_temp/frame_{frame_idx:05d}.png")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bits = extract_message(gray, locs, bit_len)
    message = bits_to_text(bits)
    print("Thông điệp trích xuất được:")
    print(message)

if __name__ == "__main__":
    main()
