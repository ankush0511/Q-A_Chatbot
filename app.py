from google.protobuf import message
import streamlit as st
from langchain.llms import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

LANGCHAIN_API_KEY=st.secrets["LANGCHAIN_API_KEY"]
GEMMINI_API_LEY=st.secrets["GEMMINI_API_LEY"]
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]

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

prompt=ChatPromptTemplate.from_messages(
    [("system","you are a helpful assistant. answer all the question to the nest of your ability in {Language}"),
     MessagesPlaceholder(variable_name="messages")
     ])
chain=prompt|model

lanres=chain.invoke({"messages":[HumanMessage(content='{response}')],
   "Language":"{language}"})

st.title("Q&A Chatbot")
st.sidebar.title("Settings")
# llm=st.title("gemmabot")
temperature=st.sidebar.slider("temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=100)
language=st.sidebar.selectbox("Select Language",("English","Spanish","German","French","Italian","Hindi"))
st.write("Go ahead")

user_inputs=st.text_input("Ask a question")
if user_inputs:
  response=chain.invoke({"messages":[HumanMessage(content=user_inputs)],"Language":language})
  st.write(response.content)
else:
  st.write("Please ask a question")
