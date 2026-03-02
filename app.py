import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser  # Ajouté
from dotenv import load_dotenv



load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")



# --- FONCTIONS PDF ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # On s'assure que le dossier est bien créé
    vector_store.save_local("faiss_index")


# --- LA CHAÎNE MODERNE (LCEL) ---
def get_conversational_chain():
    prompt_template = """
    Answer the question with detail from the context provided. 
    If the answer is not in the context, say "I don't know".\n\n
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # On crée une chaîne simple : Prompt -> Modèle -> Extraction du texte
    chain = prompt | model | StrOutputParser()
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not os.path.exists("faiss_index"):
        st.error("L'index FAISS n'existe pas. Veuillez uploader des PDFs d'abord.")
        return

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # On prépare le contexte manuellement
    context_text = "\n\n".join([d.page_content for d in docs])

    chain = get_conversational_chain()

    # Appel moderne avec .invoke()
    response = chain.invoke({"context": context_text, "question": user_question})

    st.write("Reply:", response)


# --- INTERFACE ---
def main():
    st.set_page_config(page_title="Chat PDF", page_icon="🤖")
    st.header("Chat with multiple PDFs using Gemini AI")

    user_question = st.text_input("Ask a question from your PDFs")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done & Index Saved")
                else:
                    st.error("No text found in PDFs")


if __name__ == "__main__":
    main()