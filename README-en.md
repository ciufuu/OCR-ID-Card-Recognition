# Practice Project: Text Extraction and Processing using EasyOCR and Ollama

This project was developed as part of the internship/practice program at the **Transilvania University of Brașov (UNITBV)**, Faculty of Economic Sciences and Business Administration, Economic Informatics (2nd Year).

The primary objective of the project is to implement an efficient Optical Character Recognition (OCR) pipeline to analyze and structure text extracted from various documents or images.

## 🚀 Key Features

- **Text Extraction (OCR):** Utilizes **EasyOCR** for detecting and transcribing text from various image formats, featuring multi-language support (including Romanian and English).
- **Intelligent Post-Processing:** Automatically fixes grammatical errors common in raw OCR outputs, extracts key entities (dates, names, totals), and synthesizes information from scanned documents.

## 📂 Project Structure

```text
├── data/               # Directory for test images and inputs
├── outputs/            # Saved results (extracted text, structured JSONs)
├── src/
│   ├── ocr_engine.py   # Script responsible for configuring and running EasyOCR
│   ├── nlp_processor.py# Integration with local Ollama API for text analysis
│   └── main.py         # Application entry point (orchestrating the pipeline)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ Requirements and Installation

### 1. Clone the repository

```bash
git clone https://github.com/user/proiect-easyocr-ollama.git
cd proiect-easyocr-ollama
```

### 2. Install Python dependencies

It is highly recommended to use a virtual environment (`venv`):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Core dependencies include `easyocr`, `torch`, `torchvision`, and `requests` *

## ▶️ Usage

Run the main script:

python main.py

The program will:

1. Load the ID card image
2. Preprocess the image
3. Apply OCR using EasyOCR
4. Display extracted text in structured format
5. **Input:** An image containing an invoice or a scanned document.
6. **EasyOCR:** Analyzes the image, detects characters, and returns a clean, structured file (e.g., JSON) with all relevant data fields.

## 🎓 Coordination and Evaluation

- **Institution:** Transilvania University of Brașov (UNITBV)
- **Coordinator:** Professor Maican
- **Timeline:** April 2026

## 📝 License

This repository is created solely for academic and evaluation purposes within the university framework.
