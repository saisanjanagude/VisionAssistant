# VisionAssistant AI

**VisionAssistant AI** is a multi-functional AI-powered tool designed to aid visually impaired users in understanding their surroundings through object detection, scene analysis, text extraction, and text-to-speech conversion. This application uses state-of-the-art AI technologies to provide image analysis and assist users with real-time feedback and insights.

---

## Features

- **Object Detection**: Detects and labels objects in uploaded images (using YOLO).
- **Scene Analysis**: Provides detailed descriptions of the image for visually impaired users (using Google Gemini AI).
- **Text Extraction (OCR)**: Extracts readable text from images (using Tesseract OCR).
- **Text-to-Speech (TTS)**: Converts extracted text into speech for real-time audio feedback.

---

## Tech Stack

- **Frontend**: 
  - **Streamlit** (for interactive web interface)
- **Backend**:
  - **Object Detection**: YOLO (You Only Look Once) for detecting objects in images.
  - **Text Extraction**: Tesseract OCR for Optical Character Recognition (OCR).
  - **Scene Description**: Google Gemini AI for generating descriptive text.
  - **Text-to-Speech**: pyttsx3 (for converting text to speech).
- **Other**: 
  - **Google Gemini API**: For scene analysis and generating AI-based descriptions.
  - **Python Libraries**: OpenCV, NumPy, PIL, etc.

---
