# Lane Departure Warning System (LDWS)

> **Status:** Active Development (Prototype Phase)  
> **Author:** Breogan Núñez Díaz  
> **Target Performance:** Real-time processing (~30-60 FPS) on edge devices

A lightweight, robust Advanced Driver Assistance System (ADAS) prototype built with Python and OpenCV. This system detects lane lines in video streams and evaluates vehicle position in real-time using spatial masking and geometric filtering.

---

## Project Overview

In Spain, nearly 32% of traffic accidents are caused by driver distractions leading to lane drift. This project aims to build an independent, highly efficient Lane Departure Warning System (LDWS) using classical Computer Vision and lightweight spatial logic, eliminating the need for heavy Deep Learning dependencies while maintaining real-time execution across low-power hardware.

### Key Features
- **Classical CV Pipeline:** High-speed edge detection and feature extraction without neural networks.
- **Hough Line Detection & Masking:** Spatial Region of Interest (ROI) filtering to isolate relevant road markings.
- **Slope-Based Filtering:** Discards horizontal and near-horizontal segments to eliminate noise and irrelevant markings.
- **Spatial Trap-Zone Logic:** Fast pixel-counting mechanism (`countNonZero`) inside designated alert zones to detect lane invasion.
- **Cross-Platform & Lightweight:** Optimized for ~30 FPS on standard hardware.

---

## Architecture & Pipeline

The current processing pipeline follows a structured sequence:

1. **Video Capture & Normalization:** Frame resizing to 1080p, Grayscale conversion, and Gaussian Blur to reduce high-frequency noise.
2. **Edge Detection:** Canny Edge Filter applied on the blurred grayscale frame.
3. **Spatial ROI Masking:** Masking with a polygon region to isolate the road area ahead.
4. **Line Extraction & Slope Filtering:** Probabilistic Hough Line Transform combined with slope thresholding (`|slope| > 0.5`) to keep relevant lane boundaries.
5. **Lane Departure Evaluation:** Drawing detected lines onto an alert canvas and checking line invasion within a central trap-zone.
6. **Alert Logic:** Frame-counter thresholding on positive pixel overlap to trigger warnings and discard momentary false positives.

---

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install opencv-python numpy
