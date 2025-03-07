
# Pathos: Sentiment and Emotion Detection Web Application

![Python Version](https://img.shields.io/badge/python-3.6%2B-blue?style=flat-square)
![Vue.js](https://img.shields.io/badge/Vue.js-3.x-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-red?style=flat-square)
![BERT](https://img.shields.io/badge/BERT-fine--tuned-orange?style=flat-square)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white)

<p align="center">
  <img width="256" height="171" src="app/static/img/emotions-banner.jpg">
</p>

**Pathos** is a web-based application designed to democratize access to sentiment and emotion detection, developed as part of my doctoral thesis at the **Università di Catania**. This project integrates a powerful AI model with an intuitive interface, enabling users to analyze text without programming knowledge.

## 🌟 Overview

Pathos addresses the accessibility limitations of an AI model by wrapping it in a web application built with **FastAPI** (backend) and **Vue.js** (frontend). The model detects **six emotions** (sadness, joy, love, anger, fear, surprise) and overall sentiment (positive, negative) across **three languages** (Italian, English, and Portuguese). The application uses **MongoDB** as its backend database for storing user data and feedback. It is deployable via Docker, ensuring ease of setup while accommodating the model's large size, which exceeds GitHub's storage limits.

Key features:
- **Emotion Detection**: Identifies six distinct emotions from text input.
- **Sentiment Analysis**: Classifies text as positive, or negative.
- **Multilingual Support**: Processes Italian, English, and Portuguese.
- **Secure Authentication**: Uses OAuth2 with JWTs in HTTP-only cookies.
- **Responsive Design**: Works seamlessly on desktop and mobile, with dark mode.

This project is a core component of my PhD research at Università di Catania, focusing on affective computing and human-computer interaction.

## 🚀 Features

- **Backend**: High-performance RESTful APIs with FastAPI, using MongoDB as the database.
- **Frontend**: Dynamic, reactive interface with Vue.js (client-only, served via FastAPI).
- **Model Capabilities**: Detects sadness, joy, love, anger, fear, and surprise in three languages.
- **Deployment**: Dockerized for consistent and portable setup.
- **User Feedback**: Collects input to refine model performance.

## 🔧 Installation and Setup

Pathos is designed to run via Docker due to its dependencies and model size. Follow these steps:

### Prerequisites
- Docker
- Python 3.6+ (for local development outside Docker)
- Create a `.env` file in the `app` directory with the following variables:

```plaintext
MONGO_URI=[YOUR_MONGO_URI]              # Replace with your MongoDB connection string (e.g., mongodb://localhost:27017)
DB_NAME=pathos                          # Database name, keep as is
USER_COLLECTION=users                   # Collection for user data, keep as is
FEEDBACK_COLLECTION=feedbacks           # Collection for feedback, keep as is
DB_USER=[YOUR_DB_USERNAME]              # Replace with your MongoDB admin username
DB_PWD=[YOUR_DB_PASSWORD]               # Replace with your MongoDB admin password
JWT_SECRET_KEY=[YOUR_JWT_SECRET_KEY]    # Replace with a 32-byte hex string, generate with: openssl rand -hex 32
```

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/antonioag95/pathos.git
   cd pathos
   ```

2. **Add the Model**:

   - Obtain the AI model file (details in [Model Section](#-model-and-dataset)).
   - Place it in the `app/models/` directory.

3. **Build the Docker Image**:

   ```bash
   docker build --no-cache -t pathos . --network=host
   ```

4. **Run the Container**:

   ```bash
   docker run --restart unless-stopped -d --name pathos -p 8000:8000 pathos
   ```

   - The app will be available at `http://localhost:8000`.

5. **Local Development (Optional)**:

   - Backend:

     ```bash
     cd app
     pip install -r ../requirements.txt
     uvicorn main:app --reload
     ```

   - The frontend is served directly by FastAPI using client-only Vue.js, requiring no separate setup.

## 📚 Model and Dataset

### Model
The AI model, a **BERT fine-tuned model** developed as part of my PhD at Università di Catania, detects **six emotions** (sadness, joy, love, anger, fear, surprise) and overall sentiment from text. It supports **Italian, English, and Portuguese**, leveraging advanced natural language processing techniques. Due to its size, it is not included in this repository and must be manually placed in `app/models/`. The model can be downloaded from [here](https://1drv.ms/u/c/a2bf6180319a29bd/EcUeM06-gXNDjL8Nc1HiAQ0BDbfMcz9AdSL7JEbEEvr_Lw?e=ONBFJK).

### Dataset
The dataset used for training and evaluation supports the model’s multilingual capabilities across Italian, English, and Portuguese can be find inside [this folder](https://1drv.ms/f/c/a2bf6180319a29bd/Eqdw3N--D5tJsl2edeNGOCgBwmOBRKEtQu0Q-Mjd4yJvFQ?e=aRRfiY). Details will be provided upon publication of the thesis.

## 🎓 Doctoral Thesis Context

Pathos is a key deliverable of my PhD thesis at **Università di Catania**, under the program in Complex Systems for Physical, Socio-economic and Life Sciences. It explores the deployment of affective computing models in accessible, real-world applications, emphasizing usability and multilingual support.

---

*Developed as part of my PhD journey at Università di Catania.*