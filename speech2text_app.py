import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)
    
    # Transcribe the audio file and return the result
    result = pipe(audio_file)
    return result["text"]

# Set up Gradio interface with microphone input
audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")
output_text = gr.Textbox(label="Transcription")

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
                     inputs=audio_input, outputs=output_text, 
                     title="Audio Transcription App",
                     description="Click the mic to record and transcribe your voice")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=8080)
