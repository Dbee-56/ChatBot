import streamlit as st

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
genai.configure(api_key="AIzaSyBOycvDn8d0LisDkLYAfES7ob--e1o861M")

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunck(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap=300,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = SentenceTransformer("all-mpnet-base-v2")
    # embeddings = model.encode()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = "AIzaSyBOycvDn8d0LisDkLYAfES7ob--e1o861M")
    vector_store = Chroma.from_texts(text_chunks,embeddings,persist_directory="./chroma_db")
    # vector_store.save_local("chroma_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question from the provided context, make sure to provide all the details \n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key = "AIzaSyBOycvDn8d0LisDkLYAfES7ob--e1o861M")

    promt = PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=promt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key = "AIzaSyBOycvDn8d0LisDkLYAfES7ob--e1o861M")
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    new_db.get()
    doc = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":doc,"question":user_question}
    )

    # print(response)
    st.write("Reply: ",response["output_text"])


def main():
    st.subheader("Enter any question regarding the uploaded File..")
    query = st.text_input("Enter your question..")
    if query:
        user_input(query)

    with st.sidebar:
        st.title("Chat with PDF")
        pdf = st.file_uploader("Upload your PDF and click submit",type = "pdf")
        if st.button("Submit"):
            with st.spinner("processing..."):
                raw_text = get_pdf_text(pdf)
                text_chunks = get_chunck(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        


if __name__ == "__main__":
    main()

