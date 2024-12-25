import streamlit as st
from langchain_community.llms import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model='Gemma2-9b-It')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')



prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. please response to user queries"),
    ("user", "question:{question}")
])


def generate_response(question, max_tokens, temperature):
  llm = model
  output_p = StrOutputParser()
  chain = prompt | llm | output_p
  answer = chain.invoke({'question': question})
  return answer


st.title("Q&A Chatbot")
st.sidebar.title("Settings")
# llm=st.title("gemmabot")
temperature=st.sidebar.slider("temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=100)


st.write("Go ahead")

user_inputs=st.text_input("Ask a question")
if user_inputs:
  response=generate_response(user_inputs,max_tokens,temperature)
  st.write(response)
else:
  st.write("Please ask a question")