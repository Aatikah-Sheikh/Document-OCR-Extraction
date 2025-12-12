# Document-OCR-Extraction

This repository provides a compact, end-to-end **document segmentation + OCR** pipeline in Python.  
Given an input image (e.g., a scanned page, form, or screenshot containing text), the pipeline:

1. Preprocesses the image (grayscale + thresholding + morphology)
2. Segments the image into candidate text regions (contours / connected components)
3. Runs OCR (Tesseract) on each region
4. Writes both visual and structured outputs for inspection and downstream use

---

## Repository Structure

- `segment_ocr.py`  
  Core implementation of preprocessing, region detection, OCR, and export.

- `run_demo.py`  
  Optional helper script for quick testing.

- `data/`  
  Optional location for input images.  
  A sample image may be included for demonstration purposes.

- `outputs/`  
  Output directory. This repository includes a placeholder (`.gitkeep`) to preserve the folder structure.

---

## Requirements

- Python 3.10+ (tested with Python 3.13)
- Tesseract OCR installed locally (required for OCR)

Python dependencies are listed in `requirements.txt`.

---

## Installation

Open a terminal in the repository directory (VS Code: **Terminal → New Terminal**) and run:

```bash
pip install -r requirements.txt
```

## Installing Tesseract (Windows)

Tesseract must be installed separately (it is not a pip package). A common installation path is:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```
If Tesseract is not on your PATH, set the explicit path in segment_ocr.py:
```
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
To verify Tesseract is accessible from Python:
```
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## Quick Start

Run the pipeline on an input image:
```
python segment_ocr.py --image data/sample.png --out outputs
```

## Using Your Own Image

You may run the pipeline on any .png / .jpg document image:

```
python segment_ocr.py --image "path/to/your/image.png" --out outputs
```

Tip: if your file is inside the repository, you can use relative paths:
```
python segment_ocr.py --image data/my_scan.jpg --out outputs
```

Outputs

Running the script produces outputs in the specified output directory:
```
outputs/overlay.png
The input image annotated with detected region bounding boxes and region IDs.
```
```
outputs/results.json
OCR results per region (bounding-box coordinates + extracted text).
```
```
outputs/results.csv
Spreadsheet-friendly version of the OCR results.
```
```
outputs/notes.md
Notes describing the run and pipeline behaviour (if enabled in your implementation).
```

## Interpreting Results

If overlay.png shows labelled bounding boxes around text blocks, segmentation is working as intended.

```
results.json / results.csv store OCR text for each region (region IDs correspond to those drawn in overlay.png).
```
If the overlay shows no boxes, the input likely contains no text, or segmentation parameters may need adjustment.

## Notes on Input Quality

OCR quality depends strongly on input quality. Best results are obtained with:

- high-contrast text (dark text on light background)

- adequate resolution (e.g., 200–300 DPI for scans)

- minimal skew/rotation and blur
```bash
pip install -r requirements.txt
