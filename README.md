# 🌾 Farming Video Recommendation Using Voice Recognition

This project integrates speech recognition with a recommendation system to suggest farming-related videos based on user voice commands. It leverages natural language processing and machine learning techniques to enhance accessibility and provide personalized content to users interested in agriculture.

---

## 🎯 Project Overview

The goal is to create an application that listens to user voice inputs, converts them to text, and recommends relevant farming videos accordingly. This system is especially useful for farmers and agricultural learners seeking quick access to helpful video content without typing.

---

## 🛠️ Features

- 🎤 **Voice Recognition**: Converts spoken input into text using speech recognition.
- 📽️ **Video Recommendation**: Recommends farming videos based on transcribed queries.
- 🤖 **Machine Learning Models**: Uses trained models to match speech content to relevant video topics.
- 🖥️ **User Interface**: Provides a simple interface for user interaction and query input.

---

## 📁 Project Structure

```
.
├── final_app/               # Main application directory
│   ├── app.py               # Main app script
│   └── ...                  # Supporting modules and files
├── models/                  # Trained ML models used for recommendations
│   └── ...                  
├── README.md                # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/Nahil-ElBezzari/Farming_Video_Recommendation_Using_Voice_Recognition.git
cd Farming_Video_Recommendation_Using_Voice_Recognition
```

2. **Install dependencies**

```bash
pip install speechrecognition moviepy sklearn torch
```

---

## 🚀 Usage

1. Navigate to the app directory:

```bash
cd final_app
```

2. Launch the app:

```bash
python app.py
```

3. Follow the prompts:

- Speak your farming-related query when prompted.
- The app will transcribe the audio and recommend relevant farming videos.

---

## 📚 Tech Stack

- **Speech Recognition**: Python’s `SpeechRecognition` and microphone input handling
- **Machine Learning**: Scikit-learn and PyTorch for classification or similarity matching
- **Video Recommendation**: Based on keywords or category classification
