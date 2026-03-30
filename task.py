# src/task2_segmentation.py
import cv2
import numpy as np
import csv
from pathlib import Path

INPUT = 'real_scene.jpg'
OUT_DIR = Path('results/task2_segmentation')
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESIZE_MAX = 1200
GAUSS_KSIZE = (5,5)
ADAPTIVE_BLOCK = 51
ADAPTIVE_C = 5
MIN_AREA = 100
VIS_TOP_N = 20

img = cv2.imread(INPUT)
if img is None:
    raise FileNotFoundError(f'Input image not found: {INPUT}')

h, w = img.shape[:2]
if max(h, w) > RESIZE_MAX:
    scale = RESIZE_MAX / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, GAUSS_KSIZE, 0)

_, th_global = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
th_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
_, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite(str(OUT_DIR / 'mask_global.png'), th_global)
cv2.imwrite(str(OUT_DIR / 'mask_adaptive.png'), th_adapt)
cv2.imwrite(str(OUT_DIR / 'mask_otsu.png'), th_otsu)

mask = th_otsu.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imwrite(str(OUT_DIR / 'mask_cleaned.png'), mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
measurements = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue
    perimeter = cv2.arcLength(cnt, True)
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box != 0 else 0.0
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = x + w_box//2, y + h_box//2

    eccentricity = ''
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        axes = ellipse[1]
        major_axis = max(axes)
        minor_axis = min(axes)
        if major_axis != 0:
            eccentricity = float(np.sqrt(1 - (minor_axis/major_axis)**2))
        else:
            eccentricity = 0.0

    measurements.append({
        'id': i,
        'area': float(area),
        'perimeter': float(perimeter),
        'x': int(x),
        'y': int(y),
        'w': int(w_box),
        'h': int(h_box),
        'aspect_ratio': float(aspect_ratio),
        'cx': int(cx),
        'cy': int(cy),
        'eccentricity': eccentricity
    })

measurements.sort(key=lambda m: m['area'], reverse=True)

csv_path = OUT_DIR / 'contours_measurements.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    if measurements:
        writer = csv.DictWriter(f, fieldnames=list(measurements[0].keys()))
        writer.writeheader()
        for m in measurements:
            writer.writerow(m)

for rank, m in enumerate(measurements):
    if rank >= VIS_TOP_N:
        break
    cnt = contours[m['id']]
    color = (0, 255 - min(200, int(rank*10)), min(200, int(rank*10)))
    cv2.drawContours(vis, [cnt], -1, color, 2)
    label = f"#{rank+1} A={int(m['area'])}"
    cv2.rectangle(vis, (m['x'], m['y']), (m['x']+m['w'], m['y']+m['h']), color, 1)
    cv2.putText(vis, label, (m['x'], m['y']-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite(str(OUT_DIR / 'contours_labeled.png'), vis)

print(f"Found {len(measurements)} objects (area >= {MIN_AREA}).")
print(f"Results saved to {OUT_DIR}")
