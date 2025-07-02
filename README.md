# 🧠 Disease Diagnosis & Medical Report Analyzer

This Flask-based web app enables AI-powered diagnosis from medical images (MRI, X-rays) and automatic analysis of PDF medical reports. Upload your scans or reports, and get results from deep learning models or Google Gemini AI (via the Gemini 1.5 Flash API).

---

## 🚀 Features

- 🧠 **Brain Tumor Detection** – Classify MRI scans into `Glioma`, `Meningioma`, `Pituitary`, or `No Tumor`.
- 🩺 **Pneumonia Detection** – Classify chest X-rays into `NORMAL` or `PNEUMONIA`.
- 🦴 **Bone Fracture Detection** – Detect fractures in X-ray images.
- 📄 **Medical Report Analyzer** – Upload PDF reports and get summaries using Google Gemini AI.

---

## 🖥️ Tech Stack

- **Backend**: Flask
- **Deep Learning**: TensorFlow/Keras
- **NLP**: Google Generative AI (Gemini 1.5 Flash), PyPDF2
- **Frontend**: HTML + CSS
- **Other Tools**: PIL, NumPy, Scikit-learn, transformers

---

## 📁 Folder Structure

.
├── app.py                      # Main Flask app
├── models_collect/            # Folder containing trained .h5 models
├── static/
│   ├── condition/             # Uploaded brain MRI images
│   ├── skin/                  # Uploaded chest X-ray images
│   └── bone/                  # Uploaded bone X-ray images
├── templates/
│   ├── home.html              # Homepage
│   ├── image.html             # Brain tumor result
│   ├── xray.html              # Pneumonia result
│   ├── fracture.html          # Fracture result
│   └── report_reader.html     # PDF report analyzer
└── README.md                  # This file



---

## 📦 Download Pre-trained Models

- 🧠 [Brain Tumor Classifier](https://drive.google.com/file/d/13-AS7mOFvSmYPbqGnOPfqxfzsoXzfb5Z/view?usp=sharing)
- 🩺 [Chest X-ray Pneumonia Detector](https://drive.google.com/file/d/1wkWULbRuVWPJ7d2Er-_C6wQffchkiDb2/view?usp=sharing)
- 🦴 [Bone Fracture Detector](https://drive.google.com/file/d/12uDmPR0sNU4Bj1zItrHiujl_BCwbEINJ/view?usp=sharing)

