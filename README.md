# 🎵 MoodifyAI – AI-Based Music Mood Classification and Analysis System

MoodifyAI is a machine-learning-based web application that analyzes audio files and predicts their musical mood. It provides an interactive interface to explore music mood predictions and view previous analysis results.

## ✨ What It Does

- **Music Mood Classification:** Predicts the mood of an uploaded song.
- **Audio Feature Extraction:** Extracts features such as RMS energy, chroma, tempo, and spectral characteristics using Librosa.
- **Machine Learning Prediction:** Uses a Random Forest classifier to classify music into mood categories.
- **Analysis History:** Stores previous analysis results using an SQLite database.
- **Interactive Interface:** Provides a web-based interface built with HTML, CSS, and JavaScript.

### 🎭 Supported Mood Categories
- Happy
- Sad
- Calm
- Energetic
- Angry

## 🛠️ Tools and Technologies

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Random Forest
- **Audio Processing:** Librosa, NumPy
- **Database:** SQLite
- **Dataset:** DEAM (Database for Emotion Analysis using Physiological Signals)

## 🚀 How to Run the Project

1. Install Python 3.11 or a compatible version.
2. Install the required Python libraries:
   ```bash
   pip install -r Backend/requirements.txt
   ```
3. Start the Flask backend:
   ```bash
   cd Backend
   python app.py
   ```
4. Open `Frontend/music_hub.html` using VS Code Live Server or another local web server.
5. Upload an audio file to analyze its predicted mood.

## 👥 Team Members

- **Team Member 1:** Anushka Yadav
- **Team Member 2:** Sara Khan
- **Team Member 3:** Ritika Yadav
- **Team Member 4:** Aanya Aggarwal

## 🌐 Live Demo

https://1drv.ms/v/c/6544aaf269fa2264/IQDfJM1JNFP_SZ50C4R5NlpaAd3v2__SSGr89ixLpgc1lIk?e=g0pQ0o

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. Music mood predictions are estimates and may not always be accurate.
