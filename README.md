# Lane-Departure-Warning-System
# Lane Departure Warning System (LDWS)

> **Status:** Under Active Development (Prototype Phase)  
> **Author:** Breogan Núñez Díaz
> **Target Performance:** Real-time processing (~30 FPS) on edge devices.

A lightweight, robust Advanced Driver Assistance System (ADAS) built with Python and OpenCV. This system detects lane lines in video streams and issues real-time warnings to prevent accidents caused by unintended lane departures or driver distractions.

---

## Project Overview

In Spain, nearly 32% of traffic accidents are caused by driver distractions leading to lane drift. This project aims to build an independent, highly efficient Lane Departure Warning System (LDWS) using classical Computer Vision and mathematical modeling, eliminating the need for heavy Deep Learning dependencies while maintaining real-time execution across low-power hardware.

### Key Features
- Classical CV Pipeline: High-speed edge detection and feature extraction without neural networks.
- Hough Line Detection & Masking: Spatial Region of Interest (ROI) filtering to isolate relevant road markings and ignore road noise/arrows.
- Reactive Safety Logic: Pixel-level contact point calculations to detect when the vehicle touches or crosses lane boundaries[cite: 1].
- Cross-Platform & Lightweight: Optimized for ~60 FPS on standard computing hardware[cite: 1].

---

## Architecture & Pipeline

The current processing pipeline follows a structured sequence:

1. Video Capture & Preprocessing: Frame resizing, Grayscale conversion, and Gaussian Blur to reduce high-frequency noise[cite: 1].
2. Spatial ROI Masking: Applying cv.bitwise_and with polygon masks to isolate the driving lane[cite: 1].
3. Edge Detection: Canny Edge Filter[cite: 1].
4. Line Extraction: Probabilistic Hough Line Transform[cite: 1].
5. Slope & Spatial Filtering: Filtering out horizontal segments (e.g., arrows, shadows) by line angle and minimum length.
6. Lane Departure Logic: Real-time pixel counting inside designated alert zones to trigger warnings[cite: 1].

---

## Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install opencv-python numpy
