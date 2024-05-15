import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io

# st.title("Chat Your PDFs")  # Updated title
st.set_page_config(layout="centered")
st.markdown("<h1 style='font-size:24px;'>PDF ChatBot by Ali Riaz</h1>", unsafe_allow_html=True)

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is available
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

# File Upload with user-defined name
uploaded_file = st.file_uploader("Your PDF file here", type=["pdf"])

prompt_template = """
Answer the question as detailed as possible from the provided context,
make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context",
don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n
Answer:
"""

# Additional prompts to enhance the template
prompt_template = prompt_template + """
--------------------------------------------------
Prompt Suggestions:
1. Summarize the primary theme of the context.
2. Elaborate on the crucial concepts highlighted in the context.
3. Pinpoint any supporting details or examples pertinent to the question.
4. Examine any recurring themes or patterns relevant to the question within the context.
5. Contrast differing viewpoints or elements mentioned in the context.
6. Explore the potential implications or outcomes of the information provided.
7. Assess the trustworthiness and validity of the information given.
8. Propose recommendations or advice based on the presented information.
9. Forecast likely future events or results stemming from the context.
10. Expand on the context or background information pertinent to the question.
11. Define any specialized terms or technical language used within the context.
12. Analyze any visual representations like charts or graphs in the context.
13. Highlight any restrictions or important considerations when responding to the question.
14. Examine any presuppositions or biases evident within the context.
15. Present alternate interpretations or viewpoints regarding the information provided.
16. Reflect on any moral or ethical issues raised by the context.
17. Investigate any cause-and-effect relationships identified in the context.
18. Uncover any questions or areas requiring further exploration.
19. Resolve any vague or conflicting information in the context.
20. Cite case studies or examples that demonstrate the concepts discussed in the context.
"""

# Return the enhanced prompt template
prompt_template = prompt_template + """
--------------------------------------------------
Context:\n{context}\n
Question:\n{question}\n
Answer:
"""

if uploaded_file is not None:
    st.text("File Uploaded Successfully!")

    # PDF Processing (using PyPDF2 directly)
    pdf_data = uploaded_file.read()
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    pdf_pages = pdf_reader.pages

    context = "\n\n".join(page.extract_text() for page in pdf_pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
    vector_index = FAISS.from_texts(texts, embeddings).as_retriever()

    user_question = st.text_input("Ask Anything from PDF:", "")



    if st.button("Get Answer"):
        try:
            if user_question:
                with st.spinner("Processing..."):
                    # Get Relevant Documents
                    docs = vector_index.get_relevant_documents(user_question)
                    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
                    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=google_api_key)
                    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                    st.subheader("Answer:")
                    st.write(response['output_text'])

            else:
                st.warning("Please Ask.")
        except Exception as e:
            # Handle errors
            raise HTTPException(status_code=500, detail=str(e))
