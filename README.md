# AI Body Fat Estimator with Body Part Analysis

A production-grade AI system that estimates **body fat percentage** and identifies **fat distribution** across body parts using front + side photos. Built with MediaPipe, OpenCV, and volumetric analysis.

---

## ✨ Features

- Multi-view (Front + Side) body fat estimation
- Identifies dominant fat storage area (Chest, Abdomen, Stomach, Hips, Legs)
- Real-time silhouette volumetric analysis
- Redis caching for performance
- Rate limiting for abuse protection
- Comprehensive test coverage

---

## 🛠 Tech Stack

- **Framework**: FastAPI
- **Computer Vision**: MediaPipe Selfie Segmentation + OpenCV
- **Caching**: Redis
- **Rate Limiting**: SlowAPI
- **Testing**: Pytest + AsyncClient

---

## 🚀 Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/ai-body-fat-estimator.git
cd ai-body-fat-estimator
