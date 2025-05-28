# Medicinal-Plants-Analysis


# 🌿 Medicinal Plant Analysis using Xception

This project is a full-stack web application designed to analyze and classify medicinal plant leaves using a deep learning model based on the **Xception** architecture. It aims to identify the plant species and provide insights into their medicinal properties.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Model Information](#model-information)


---

## 🧾 Introduction

Medicinal plants have been used for centuries in traditional healthcare. Identifying them correctly is crucial for their effective use. This project leverages a pre-trained **Xception** model fine-tuned on a dataset of medicinal plants to perform image classification and display medicinal properties of the identified plant.

---

## 🧰 Tech Stack

### 🔙 Backend
- **Python 3.x**
- **Flask** – lightweight web framework
- **TensorFlow/Keras** – for model loading and prediction
- **Xception** – pre-trained deep learning model
- **OpenCV** / **Pillow** – for image preprocessing

### 🎨 Frontend
- **React.js** – interactive UI
- **Tailwind CSS** – utility-first CSS framework
- **Vite** – fast build tool and dev server

---

## 📁 Project Structure

MedicinalPlantAnalysis/
│<br>
├── backend/<br>
│ ├── app.py # Flask backend<br>
│ ├── requirement.txt # Python dependencies<br>
│ ├── test.ipynb # Jupyter notebook for testing<br>
│ ├── cnnmodel/ # Model files<br>
│ ├── temp_uploads/ # Temporary image uploads<br>
│ └── venv/ # Virtual environment<br>
│<br>
├── frontend/<br>
│ ├── public/ # Static files<br>
│ ├── src/ # React source code<br>
│ ├── index.html # Main HTML file<br>
│ ├── tailwind.config.js # Tailwind config<br>
│ └── vite.config.js # Vite config<br>
│<br>
├── ML Model/<br>
│ └── MedicinalPlant.py # Model building and evaluation<br>
│<br>
└── README.md<br>


---

## ✨ Features

- Upload a plant leaf image and classify it into a medicinal species.
- Provides scientific and common names.
- Displays medicinal uses and benefits.
- Responsive web interface.
- Backend API with image preprocessing and classification logic.

---

## 🛠️ Installation

### 1. Clone the Repository

bash
git clone https://github.com/yourusername/medicinal-plant-analysis.git
cd medicinal-plant-analysis 
## 2. Setup Backend

To set up the Python backend environment:

bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirement.txt


---


## 3. Setup Frontend

To install dependencies for the frontend (React):


cd ../frontend
npm install



---



## ▶️ Run Backend

To start the backend server:


cd backend
python app.py


---


## ▶️ Run Frontend

To start the frontend development server:

cd frontend
npm run dev

---

Model Information-- <br>
Model Architecture: Xception (Extreme Inception)<br>
Input: RGB image of a plant leaf<br>
Output: Predicted plant class and medicinal description<br>
Dataset: Custom dataset of medicinal leaves<br>
Performance: Achieved high classification accuracy on validation data<br>


