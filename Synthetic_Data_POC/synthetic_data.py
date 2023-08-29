import streamlit as st
import langchain.llms
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain 
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os

llm_list = ['chat gpt','Falcon','FLAN']
chosen_llm = st.selectbox(label = "Choose a model", options = llm_list)

if (chosen_llm == 'chat gpt'):
    os.environ['OPENAI_API_KEY']
    llm = OpenAI(temperature=0.9) 
elif (chosen_llm == 'Falcon'):
    hub_llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b")
elif (chosen_llm == 'FLAN'):
    hub_llm = HuggingFaceHub(repo_id="google/flan-t5-xl")

if(chosen_llm == 'chat gpt'):
    prompt = PromptTemplate(
    input_variables=["info_template"],
    template="You specialize in creating data and have been tasked with generating example of data using the provided request in json format : {info_template}."    
    )   
else:
    prompt = PromptTemplate(
    input_variables=["info_template"],
    template=" : {info_template}."    
    ) 

input = st.text_input("input the kind of data you require and the format")

if st.button("submit"):
     with st.spinner("Processing"):
        hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
        res = hub_chain.run(input)
        st.write(res)
