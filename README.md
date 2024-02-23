# COCO to DOTA Annotation Converter

This Python script converts image annotations from the COCO format (commonly used in computer vision tasks) to the DOTA format for object detection. The DOTA format supports rotated or oriented bounding boxes (OBB).

## Usage

1. **Dependencies:**
   - Requires `json`, `numpy`, and `scipy` libraries.

2. **Functionality:**
   - Defines a function `minimum_bounding_rectangle` to calculate the minimum bounding rectangle for a set of points.
   - Iterates through COCO format JSON file containing image annotations.
   - Converts each annotation to DOTA format, considering rotated bounding boxes.
   - Writes DOTA formatted annotations to corresponding label files.

3. **Output:**
   - DOTA format label files are generated in the specified destination directory.

4. **Note:**
   - Checks and corrections for negative bounding box coordinates are included.
   - A comment suggests a potential future improvement or check.

## How to Run

1. Modify the `cocojson` and `destLabels` variables to point to your COCO JSON file and destination directory.
2. Execute the script.

```bash
python coco_to_dota_converter.py
