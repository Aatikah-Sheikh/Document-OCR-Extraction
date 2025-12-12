import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
import pytesseract

@dataclass
class RegionResult:
    region_id: int
    x: int
    y: int
    w: int
    h: int
    text: str

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    # Adaptive threshold works well for uneven lighting
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 15
    )

    # Connect characters into blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph

def find_regions(mask: np.ndarray) -> list[tuple[int,int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = mask.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        # Filter: remove tiny noise + overly huge regions
        area = w * h
        if area < 800:         # tune
            continue
        if area > 0.90 * (W*H):
            continue

        # Filter: remove very thin lines
        if h < 15 or w < 30:
            continue

        boxes.append((x, y, w, h))

    # Sort top-to-bottom then left-to-right
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def ocr_region(img_bgr: np.ndarray, box: tuple[int,int,int,int]) -> str:
    x, y, w, h = box
    roi = img_bgr[y:y+h, x:x+w]

    # OCR expects black text on white background (often better)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = "--oem 3 --psm 6"  # general block of text
    text = pytesseract.image_to_string(gray, config=config)
    return text.strip()

def draw_overlay(img_bgr: np.ndarray, boxes: list[tuple[int,int,int,int]]) -> np.ndarray:
    out = img_bgr.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(out, f"{i}", (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

def run(image_path: str, out_dir: str = "outputs") -> None:
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = preprocess(img)
    boxes = find_regions(mask)

    results: list[RegionResult] = []
    for idx, box in enumerate(boxes, start=1):
        text = ocr_region(img, box)
        x, y, w, h = box
        results.append(RegionResult(idx, x, y, w, h, text))

    overlay = draw_overlay(img, boxes)
    cv2.imwrite(str(outp / "overlay.png"), overlay)

    with open(outp / "results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # Simple CSV
    with open(outp / "results.csv", "w", encoding="utf-8") as f:
        f.write("region_id,x,y,w,h,text\n")
        for r in results:
            safe = r.text.replace("\n", " ").replace('"', "'")
            f.write(f'{r.region_id},{r.x},{r.y},{r.w},{r.h},"{safe}"\n')

    # Notes template
    notes = outp / "notes.md"
    if not notes.exists():
        notes.write_text(
            "- Segmentation method: adaptive threshold + morphology + contours\n"
            "- OCR engine: Tesseract (pytesseract)\n"
            "- Known limitations: skewed scans, low resolution, handwritten text\n",
            encoding="utf-8"
        )

    print(f"Done. Wrote: {outp/'overlay.png'}, {outp/'results.json'}, {outp/'results.csv'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image (png/jpg)")
    ap.add_argument("--out", default="outputs", help="Output directory")
    args = ap.parse_args()
    run(args.image, args.out)
