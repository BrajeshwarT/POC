import streamlit as st
import pandas as pd
import numpy as np
import langchain.llms
import matplotlib.pyplot as plt  # Import matplotlib
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
import os
import regex as re
import json

def cnvt(res):
    res = re.sub("[\n]", "", res)
    res = res.replace(" ", "")
    res_list = json.loads(res)
    return res_list

def dtfr(inpt):
    df = pd.DataFrame.from_dict(inpt)
    return df

def main():
    load_dotenv(find_dotenv())
    HUGGINGFACEHUB_API_TOKEN = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    llm_list = ['chat gpt', 'Falcon', 'FLAN']
    chosen_llm = st.selectbox(label="Choose a model", options=llm_list)

    if chosen_llm == 'chat gpt':
        hub_llm = OpenAI(temperature=1, max_tokens=4000)
    elif chosen_llm == 'Falcon':
        hub_llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b")
    elif chosen_llm == 'FLAN':
        hub_llm = HuggingFaceHub(repo_id="google/flan-t5-xl")

    if chosen_llm == 'chat gpt':
        prompt = PromptTemplate(
            input_variables=["info_template"],
            template="You specialize in creating synthetic data in json format, and have been tasked with generating 10 examples of synthetic data using the provided request in json format. only create data in json format(key:value)  : {info_template}"
        )
    else:
        prompt = PromptTemplate(
            input_variables=["info_template"],
            template="Create data : {info_template}."
        )

    input_data = st.text_input("Input the kind of data you require:")
    template = st.text_input("Columns/template of the dataset:")

    # Customize Data Generation
    data_distribution = st.selectbox("Choose Data Distribution", ["Normal", "Uniform"])
    noise_level = st.slider("Noise Level", min_value=0.0, max_value=1.0, step=0.1, value=0.0)

    st.write("For now, the outputs have been capped/limited to 10 rows to prevent misuse.")

    if st.button("Submit"):
        with st.spinner("Processing"):
            hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

            if data_distribution == "Normal":
                input_data_1 = input_data +" having data with normal distribution and noise level "+str(noise_level)
            elif data_distribution == "Uniform":
                input_data_1 = input_data +" having data with uniform distribution and noise level "+str(noise_level)
            res = hub_chain.run(input_data_1 + template + " in json format")
            a = cnvt(res)
            b = dtfr(a)

            # Convert JSON data to DataFrame
            df = pd.DataFrame.from_dict(b)

            # Apply Data Customization only to numeric columns
            # numeric_cols = df.select_dtypes(include=[np.number]).columns
            # if data_distribution == "Normal":
            #     df[numeric_cols] += noise_level * df[numeric_cols].std() * np.random.randn(len(df))
            # elif data_distribution == "Uniform":
            #     df[numeric_cols] += noise_level * (df[numeric_cols].max() - df[numeric_cols].min()) * np.random.rand(len(df))

            st.write(df)
            
            # Export to CSV
            csv_filename = 'myfile.csv'
            df.to_csv(csv_filename, index=False)
            st.download_button('Download CSV', csv_filename, file_name='myfile.csv', key='csv-download')

            # Export to JSON
            json_filename = 'myfile.json'
            with open(json_filename, 'w') as json_file:
                json.dump(a, json_file, indent=4)
            st.download_button('Download JSON', json_filename, file_name='myfile.json', key='json-download')

            st.write("### Data Visualization")
            st.write("Bar chart of data distribution")
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # if data_distribution == "Normal":
            #     fig, ax = plt.subplots()
            #     ax.bar(numeric_cols, df.mean(), yerr=df.std())
            #     ax.set_xlabel("Columns")
            #     ax.set_ylabel("Mean")
            #     ax.set_title("Mean Values with Standard Deviation")
            #     st.pyplot(fig)  # Display the matplotlib figure

            # elif data_distribution == "Uniform":
            #     fig, ax = plt.subplots()
            #     ax.bar(numeric_cols, df.max() - df.min())
            #     ax.set_xlabel("Columns")
            #     ax.set_ylabel("Range")
            #     ax.set_title("Column Ranges")
            #     st.pyplot(fig)  # Display the matplotlib figure

if __name__ == '__main__':
    main()
