import numpy as np
import os

fp = r'd:/experiment/SingleTrack_Project/results/traditional/GOT-10k_Val_000146/color/trajectory.csv'
with open(fp, 'rb') as f:
    text = f.read().decode('utf-8-sig')

lines = text.strip().split('\n')
print(f'总{len(lines)-1}帧')

prev = None
changes = 0
for i, line in enumerate(lines[1:]):
    parts = line.strip().split(',')
    x = int(float(parts[1]))
    y = int(float(parts[2]))
    w = int(float(parts[3]))
    h = int(float(parts[4]))
    curr = (x, y, w, h)
    if prev is not None and curr != prev:
        changes += 1
        if changes <= 10:
            print(f'  帧{i}: ({x},{y},{w},{h}) 变化!')
    prev = curr

first = lines[1].strip().split(',')
last = lines[-1].strip().split(',')
print(f'\n首次: ({int(float(first[1]))},{int(float(first[2]))},{int(float(first[3]))},{int(float(first[4]))})')
print(f'末帧: ({int(float(last[1]))},{int(float(last[2]))},{int(float(last[3]))},{int(float(last[4]))})')
print(f'总共变化次数: {changes}')
print(f'X唯一值: {len(set(int(float(l.split(\",\")[1])) for l in lines[1:]))}')