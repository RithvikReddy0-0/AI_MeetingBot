# 🎙️ Audio Key Point Extractor using Whisper & WatsonX

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![Gradio App](https://img.shields.io/badge/Built%20with-Gradio-orange)](https://gradio.app)
[![WatsonX](https://img.shields.io/badge/Powered%20by-WatsonX-blueviolet)](https://www.ibm.com/watsonx)

This project is a **speech-to-insight Gradio app** that:
- Transcribes uploaded audio files using **OpenAI Whisper**
- Extracts and summarizes **key points** using **IBM WatsonX LLM (LLaMA2-70B)**

Ideal for use cases like:
> 📋 Meeting minutes, 🎤 Interview processing, 🎧 Lecture summarization, or any audio-to-summary task.

---

## 🚀 Demo

> Upload any `.wav`, `.mp3`, or `.ogg` audio file.  
> The app will return a clean, detailed summary of the content.

<p align="center">
  <img src="https://github.com/your-username/audio-keypoint-extractor/assets/demo.gif" alt="Demo GIF" width="600"/>
</p>

---

## 🧠 Tech Stack

| Component        | Description                              |
|------------------|------------------------------------------|
| **Python**       | Backend programming                      |
| **Gradio**       | Frontend interface for file input/output |
| **Whisper**      | OpenAI’s automatic speech recognition    |
| **WatsonX**      | IBM’s enterprise-grade LLM (LLaMA2)      |
| **LangChain**    | Prompt orchestration                     |

---

## 🛠️ Installation

1. **Clone the repo**

    ```bash
    git clone https://github.com/RithvikReddy0-0/AI_MeetingBot.git
    cd AI_MeetingBot

2. **Clone the repo**

    ```txt
      torch
      transformers
      gradio
      langchain
      ibm-watson-machine-learning
      Ensure PyTorch and CUDA are properly installed for GPU acceleration.

3. **Set up**
4. **Run the app**
     ```bash
     python app.py

then got to http://localhost:7860

---

