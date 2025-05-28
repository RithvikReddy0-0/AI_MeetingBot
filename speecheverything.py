import torch
import os
import gradio as gr
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# -------------------------------
# Configuration
# -------------------------------

# Securely load credentials from environment variables
watsonx_api_key = os.getenv("WATSONX_API_KEY", "your_api_key_here")
watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "your_project_id_here")
watsonx_model_id = "meta-llama/llama-2-70b-chat"  # You can replace with another model if needed

# -------------------------------
# Initialize IBM WatsonX LLM
# -------------------------------

model = Model(
    model_id=watsonx_model_id,
    params={
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.7,
    },
    credentials={
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": watsonx_api_key,
    },
    project_id=watsonx_project_id,
)

llm = WatsonxLLM(model=model)

# -------------------------------
# Prompt Template for Key Point Extraction
# -------------------------------

template = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=template
)

prompt_chain = LLMChain(llm=llm, prompt=prompt_template)

# -------------------------------
# Speech to Text + Key Point Extraction
# -------------------------------

def transcript_audio(audio_file):
    try:
        # Initialize Whisper model
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)
        
        # Perform transcription
        transcript = pipe(audio_file, batch_size=8)["text"]
        
        # Extract key points
        result = prompt_chain.run(transcript)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# Gradio UI Setup
# -------------------------------

audio_input = gr.Audio(sources="upload", type="filepath", label="Upload Audio File")
output_text = gr.Textbox(label="Extracted Key Points")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="🎤 Audio Key Point Extractor",
    description="Upload an audio file. It will transcribe and extract detailed key points using OpenAI Whisper and IBM WatsonX LLM."
)

# -------------------------------
# Launch App
# -------------------------------

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
