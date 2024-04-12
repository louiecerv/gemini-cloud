import streamlit as st
import os
import keras
import keras_nlp

KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')
os.environ["KERAS_BACKEND"] = "jax"  # Or "tensorflow" or "torch".
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

def app():
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

    gemma_lm.summary()

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
        bot_response = gemma_lm.generate(user_input, max_length=2048)

        # Add bot response to chat history
        chat_history.append({"speaker": "Gemma", "message": bot_response})

    # Display chat history
    for message in chat_history:
        st.write(f"{message['speaker']}: {message['message']}")

#run the app
if __name__ == "__main__":
  app()
