import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


PDF_DIRECTORY = "pdfs"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text_from_directory(directory_path):
    text = ""
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            try:
                pdf_reader = PdfReader(pdf_path)
            except Exception as e:
                st.error(f"PDF okunurken hata oluştu: {str(e)}")
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

def texts_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    1. Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
    2. Görevin, yalnızca sağlanan metin alıntılarını kullanarak Dokuz Eylül Üniversitesi adına cevap vermektir.
    3. Yanıtı oluştururken şu kurallara dikkat et:
   - Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
   - Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
    4. Yanıtı, Türkçe dilinde ve anlaşılır bir şekilde ver.
    5. Kullanıcıya her zaman yardımcı olmaya çalış, ancak mevcut bilgilere dayanmayan yanıtlardan kaçın.
    6. Eğer "Sen kimsin" diye bir soru gelirse "Ben Dokuz Eylül Üniversitesinin asistan botuyum , amacım sizlere üniversite hakkında bilgi sağlamak ! Size nasıl yardımcı olabilirim ?" diye cevap ver.
    Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.
    Context: \n {context}?\n
    Kullanıcı Sorusu: \n{question}\n

    Yanıt:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro" , temperature = 0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(input):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(input)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":input}, return_only_outputs=True

    )

    print(response)
    st.write("Cevap: ", response["output_text"])



def main():
    st.set_page_config(page_title="DEUbot", page_icon="🏫", layout="centered")
    st.header("Merhaba ben DEUbot 🤖")
    st.subheader("Üniversitemiz hakkında ne öğrenmek istersin?")

    # CSS ile sadece soru kutucuğunu ortalamak
    st.markdown("""
        <style>
        .centered-input {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 3vh;
        }
        </style>
        """, unsafe_allow_html=True)

    # Ortalanmış text input
    st.markdown('<div class="centered-input">', unsafe_allow_html=True)
    user_question = st.text_input("Sorunuzu Giriniz : ")
    st.markdown('</div>', unsafe_allow_html=True)

    if user_question:
        # user_input fonksiyonunuzu çağırabilirsiniz.
        with st.spinner("Sorunuz işleniyor, lütfen bekleyin..."):
            user_input(user_question)

if __name__ == "__main__":
    main()