# Practice Project: Text Extraction and Processing using EasyOCR 🪪

![Python](https://img.shields.io/badge/Python-3.11.9-blue.svg)
![OCR](https://img.shields.io/badge/OCR-EasyOCR-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-orange.svg)


This project was developed as part of the internship/practice program at the **Transilvania University of Brașov (UNITBV)**, Faculty of Economic Sciences and Business Administration, Economic Informatics (2nd Year).

The primary objective of the project is to implement an efficient Optical Character Recognition (OCR) pipeline to analyze and structure text extracted from various documents or images.


## 🚀 Key Features


- **Text Extraction (OCR):** Utilizes **EasyOCR** for detecting and transcribing text from various image formats, featuring multi-language support (including Romanian and English).
- **Intelligent Post-Processing:** Automatically fixes grammatical errors common in raw OCR outputs, extracts key entities (dates, names, totals), and synthesizes information from scanned documents.



## 📂 Project Structure

```text
├── images/             # Directory for input ID cards and images to be processed
├── README.md           # Project documentation 
├── main.py             # Main Python script responsible for running EasyOCR and the processing pipeline
├── requirements.txt    # Project dependencies
└── rezultate.json      # JSON output file containing the extracted data results
```

## 🛠️ Tech Stack

1. Python 3.11.9
1. EasyOCR
1. OpenCV
1. NumPy

## 🛠️ Requirements and Installation

### 1. Clone the repository

```bash
git clone https://github.com/ciufuu/ocr-id-card-recognition.git
cd ocr-id-card-recognition
```

### 2. Install Python dependencies

It is highly recommended to use a virtual environment (`venv`):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Core dependencies include *easyocr*, *torch* and *OpenCV* *

## ▶️ Usage

Run the main script:

python main.py

The program will:

1. Load the ID card image
2. Preprocess the image
3. Apply OCR using EasyOCR
4. Display extracted text in structured format

## ⚙️ How it works

1. Image Preprocessing

- Grayscale conversion
- Noise reduction
- Contrast enhancement

2. OCR Processing

- EasyOCR model detects text regions
- Extracts raw text from image

3. Post-processing

- Cleans and formats extracted text
- Maps values to fields (Name, CNP, etc.)

## 🎓 Coordination and Evaluation

- **Institution:** Transilvania University of Brașov (UNITBV)
- **Coordinator:** Professor Maican
- **Timeline:** April 2026

## 📝 License

This repository is created solely for academic and evaluation purposes within the university framework.

## ⭐ Notes

- Results depend on image quality
- Works best on clear scanned documents
- Can be extended to other document types
