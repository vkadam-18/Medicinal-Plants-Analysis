# gemini.py

import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyAVmOOxxCLbcqDFETE2NQhYvzW86qJ7Jyo")  # move to .env later for safety

# Create Gemini model instance
model = genai.GenerativeModel('gemini-1.5-flash-8b-exp-0924')

def ask_gemini(plant_name: str, question: str) -> str:
    """
    Old function for chat-based question answering.
    """
    prompt = f"The plant is {plant_name}. User's question: {question}"
    response = model.generate_content(prompt)
    return response.text

def generate_plant_info(plant_name: str) -> str:
    """
    New function: Generate a short paragraph about the plant.
    """
    prompt = f"Give me 5 to 10 short bullet points about the medicinal plant {plant_name}. Keep each point concise."
    response = model.generate_content(prompt)
    return response.text
