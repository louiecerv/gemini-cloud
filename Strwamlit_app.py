import streamlit as st
from gemma import Gemma

def app():

  # Load the pre-trained model
  model = Gemma.load("gemma-large")

  # Create a new chatbot instance
  chatbot = Gemma(model)

  # Define a function to get user input
  def get_user_input():
      return input("You: ")

  # Define a function to get chatbot response
  def get_chatbot_response(user_input):
      return chatbot.generate_response(user_input)

  # Main loop for the chatbot conversation
  while True:
      user_input = get_user_input()
      
      if user_input.lower() == "exit":
          break
      
      chatbot_response = get_chatbot_response(user_input)
      st.write("Chatbot:", chatbot_response)

#run the app
if __name__ == "__main__":
  app()
