# Drone Human Detection & Counting System

Computer vision pipeline that detects humans and cars in aerial drone imagery, counts humans per image, displays bounding boxes, and tracks objects across video frames. Built for the Antlings AI/ML internship technical assessment.

## Results
| Metric | Value |
|---|---|
| mAP@50 (overall) | **0.258** |
| mAP@50-95 (overall) | **0.143** |
| Precision | **0.384** |
| Recall | **0.293** |
| Inference speed | **1.5 ms/img (~660 FPS)** on Tesla T4 |
| Training time | **40 min for 15 epochs** |

### Per-class mAP@50 (highlights)
- **car** — 0.686 (strong detector)
- **pedestrian** — 0.263
- **people** — 0.196
- **van** — 0.290
- **bus** — 0.358

## Approach
- **Model:** YOLOv8n (Ultralytics) — 3.0M parameters, single-shot detector
- **Initialization:** Transfer learning from COCO pretrained weights
- **Dataset:** VisDrone2019-DET — 6,471 train / 548 val / 1,610 test images
- **Training:** 15 epochs, image size 640, batch 16, AdamW optimizer, on Google Colab T4 GPU
- **Target classes for counting:** humans = pedestrian (0) + people (1); cars = class 3
- **Tracking (bonus):** ByteTrack via Ultralytics integration

## Pipeline
1. **Dataset understanding** — VisDrone has 10 classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor) captured from drones over Chinese urban/suburban scenes. Annotations are in YOLO format (class cx cy w h, normalized).
2. **Preprocessing** — handled by Ultralytics: letterbox resize to 640×640, normalization, and augmentation (mosaic, HSV jitter, h-flip, scale).
3. **Training** — YOLOv8n fine-tuned from COCO weights for 15 epochs with early stopping (patience=5).
4. **Detection & counting** — filter detections for human classes (0,1) and car class (3); count is `len(filtered_detections)` per image. Confidence threshold 0.25.
5. **Tracking (bonus)** — ByteTrack assigns persistent IDs across frames via IoU matching + Kalman motion prediction.
6. **Evaluation** — mAP@50, mAP@50-95, precision, recall on validation split, plus per-class breakdown.

## Files
- `Technical_Assessment.ipynb` — full pipeline, runnable in Colab
- `visdrone_outputs/` — all generated outputs (visualizations, predictions, tracking video, metrics, model weights)

## How to run
1. Open `Technical_Assessment.ipynb` in Google Colab with T4 GPU runtime
2. Mount Google Drive containing the VisDrone dataset (or use the Kaggle API)
3. Run cells top to bottom

## Challenges noticed in the dataset
- **Tiny objects** — pedestrians from high-altitude drone shots are often under 20 pixels tall, the hardest case for detectors at 640px input resolution
- **Top-down perspective** — COCO-pretrained features assume horizontal viewpoint; the model has to relearn aerial geometry
- **Dense crowds** — overlapping people in markets and crosswalks cause merged or duplicate boxes
- **Class imbalance** — cars and pedestrians dominate; rare classes (awning-tricycle, bus) get low per-class mAP
- **Varying lighting and altitude** — day/night, different flying heights, motion blur

## Strengths
- Single end-to-end pipeline from raw data through evaluation in one notebook
- Strong car-detection performance (mAP@50 = 0.686)
- Real-time inference speed (~660 FPS) — easily deployable
- Clear visualizations at every stage of the pipeline
- Tracking-enabled for video inputs

## Limitations
- Pedestrian recall is limited by tiny-object size at 640 px input
- Dense crowds cause merged boxes when people heavily overlap
- YOLOv8n trades accuracy for speed; YOLOv8s or m would improve mAP at 2-3× training cost
- Counting without tracking double-counts the same person across video frames; ByteTrack mitigates this but struggles with heavy occlusion
- Rare classes have low per-class mAP due to class imbalance

## What I would improve with more time
- Train YOLOv8s or YOLOv8m for higher overall mAP
- Increase input image size to 1280 to better detect tiny objects
- Add tiled inference (sliding-window crops) for very small objects
- Tune per-class confidence thresholds

## Tech stack
Ultralytics YOLOv8 · PyTorch · OpenCV · Python 3.12 · Google Colab T4
