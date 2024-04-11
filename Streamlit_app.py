import streamlit as st
import streamlit as st
from transformers import pipeline
import os

# Replace with your own Hugging Face access token (get one from https://huggingface.co/docs/hub/en/security-tokens)
#huggingface_token = st.secrets["HUGGINGFACE_TOKEN"]
huggingface_token = os.getenv("GEMMA_TOKEN")

# Define function to load Gemma model
@st.cache(allow_output_mutation=True)
def load_gemma():
    return pipeline("text-davinci-003", model="google/gemmia", task="dialogue", device=0,  # Adjust device for GPU usage (if available)
                    auth_token=huggingface_token)

def app():

    # Load Gemma model
    gemma = load_gemma()

    # Initialize chat history
    chat_history = []

    st.title("Chat with Gemma")

    # Text input for user message
    user_input = st.text_input("You:")

    # Button to submit message
    if st.button("Send"):
        # Add user message to chat history
        chat_history.append({"speaker": "User", "message": user_input})

        # Generate response from Gemma
        bot_response = gemma(prompt=user_input, max_length=1000,  # Adjust max_length for longer responses (be mindful of usage limits)
                            do_sample=True, top_k=50, top_p=0.9)["generated_text"][0]

        # Add bot response to chat history
        chat_history.append({"speaker": "Gemma", "message": bot_response})

    # Display chat history
    for message in chat_history:
        st.write(f"{message['speaker']}: {message['message']}")

#run the app
if __name__ == "__main__":
  app()
