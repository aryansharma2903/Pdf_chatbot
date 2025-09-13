import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
import uuid
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from base64 import b64decode
from unstructured.documents.elements import Image, Table, CompositeElement

# ====== NEW IMPORTS ======
import fitz  # PyMuPDF for image extraction
import base64
# ====== END NEW IMPORTS ======

# Set page config
st.set_page_config(
    page_title="Multi-Modal RAG PDF ChatBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class^='css'] {
        background-color: #f0f2f5 !important;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #1f2937;
        margin-bottom: 2rem;
        letter-spacing: -0.05rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .chat-message {
        padding: 1.1rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        font-weight: 500;
        word-break:break-word;
        border: none !important;
    }
    .user-message {
        background-color: #ffffff;
        color: #1f2937;
        border-left: 4px solid #4f46e5;
    }
    .bot-message {
        background-color: #f9fafb;
        color: #374151;
        border-left: 4px solid #f97316;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding-top: 2rem;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    .upload-section {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        background-color: #fafbfd;
    }
    .stChatInputContainer {
        background: #ffffff;
        border-top: 1px solid #e5e7eb;
    }
    .chat-message strong {
        color: #111;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

@st.cache_resource
def create_summarize_chain():
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.
    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.
    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    return {"element": lambda x: x} | prompt | model | StrOutputParser()

@st.cache_resource
def create_image_summary_chain():
    prompt_template = """Describe the image in detail. For context,
      the image is part of a research paper explaining the transformers
      architecture. Be specific about graphs, such as bar plots."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | ChatMistralAI(model="pixtral-large-2411") | StrOutputParser()

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF using PyMuPDF"""
    images_base64 = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    encoded_img = base64.b64encode(image_bytes).decode("utf-8")
                    images_base64.append(encoded_img)
                except Exception as e:
                    st.warning(f"Could not extract image {img_index} from page {page_num}: {str(e)}")
                    continue
        doc.close()
    except Exception as e:
        st.error(f"Error extracting images from PDF: {str(e)}")
    return images_base64

class ImageElement:
    """Custom class to store image data"""
    def __init__(self, image_base64):
        self.metadata = {"image_base64": image_base64}
        self.text = ""

def process_pdf(file_path):
    with st.spinner("Processing PDF... This may take a few minutes."):
        try:
            # Extract images first
            images_base64 = extract_images_from_pdf(file_path)
            
            # Partition PDF
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                chunking_strategy="basic",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )
            
            texts, tables = [], []
            for chunk in chunks:
                if isinstance(chunk, Table):
                    tables.append(chunk)
                elif hasattr(chunk, "text") and isinstance(chunk.text, str) and chunk.text.strip():
                    texts.append(chunk)
            
            # Create custom image elements instead of using unstructured Image class
            images = [ImageElement(img_b64) for img_b64 in images_base64]
            
            st.success(f"Extracted {len(texts)} text chunks, {len(tables)} tables, and {len(images)} images!")
            return texts, tables, images
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return [], [], []

def create_summaries(texts, tables, images):
    with st.spinner("Creating summaries..."):
        try:
            summarize_chain = create_summarize_chain()
            image_chain = create_image_summary_chain()
            
            # Process text summaries
            text_summaries = []
            if texts:
                try:
                    text_summaries = summarize_chain.batch([t.text for t in texts], {"max_concurrency": 1})
                except Exception as e:
                    st.warning(f"Error creating text summaries: {str(e)}")
                    text_summaries = []
            
            # Process table summaries
            table_summaries = []
            if tables:
                try:
                    tables_html = [table.metadata.text_as_html for table in tables if hasattr(table.metadata, 'text_as_html')]
                    if tables_html:
                        table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 1})
                except Exception as e:
                    st.warning(f"Error creating table summaries: {str(e)}")
                    table_summaries = []
            
            # Process image summaries
            image_summaries = []
            if images:
                try:
                    image_summaries = image_chain.batch([img.metadata["image_base64"] for img in images])
                except Exception as e:
                    st.warning(f"Error creating image summaries: {str(e)}")
                    image_summaries = []
            
            return text_summaries, table_summaries, image_summaries
            
        except Exception as e:
            st.error(f"Error creating summaries: {str(e)}")
            return [], [], []

def create_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries):
    try:
        vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=MistralAIEmbeddings())
        store = InMemoryStore()
        id_key = "doc_id"
        retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
        
        # Add text documents
        if texts and text_summaries:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            summary_texts = [
                Document(page_content=summary, metadata={id_key: doc_ids[i]})
                for i, summary in enumerate(text_summaries)
                if summary and i < len(texts)
            ]
            if summary_texts:
                retriever.vectorstore.add_documents(summary_texts)
                retriever.docstore.mset([
                    (doc_ids[i], texts[i].text)
                    for i, summary in enumerate(text_summaries)
                    if summary and i < len(texts)
                ])
        
        # Add table documents
        if tables and table_summaries:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            summary_tables = [
                Document(page_content=summary, metadata={id_key: table_ids[i]})
                for i, summary in enumerate(table_summaries)
                if summary and i < len(tables) and hasattr(tables[i].metadata, 'text_as_html')
            ]
            if summary_tables:
                retriever.vectorstore.add_documents(summary_tables)
                retriever.docstore.mset([
                    (table_ids[i], tables[i].metadata.text_as_html)
                    for i, summary in enumerate(table_summaries)
                    if summary and i < len(tables) and hasattr(tables[i].metadata, 'text_as_html')
                ])
        
        # Add image documents
        if images and image_summaries:
            img_ids = [str(uuid.uuid4()) for _ in images]
            summary_img = [
                Document(page_content=summary, metadata={id_key: img_ids[i]})
                for i, summary in enumerate(image_summaries)
                if summary and i < len(images)
            ]
            if summary_img:
                retriever.vectorstore.add_documents(summary_img)
                retriever.docstore.mset([
                    (img_ids[i], images[i].metadata["image_base64"])
                    for i, summary in enumerate(image_summaries)
                    if summary and i < len(images)
                ])
        
        # Check if any documents were added
        if not (text_summaries or table_summaries or image_summaries):
            st.error("No valid chunks with summaries found for retrieval. PDF may be text-only or partitioning failed.")
            return None
            
        return retriever
        
    except Exception as e:
        st.error(f"Error creating retriever: {str(e)}")
        return None

def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        if isinstance(doc, str):
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        else:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    
    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            if hasattr(text_element, 'text'):
                context_text += text_element.text + "\n\n"
            else:
                context_text += str(text_element) + "\n\n"
    
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """
    
    prompt_content = [{"type": "text", "text": prompt_template}]
    
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
    
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

# Main UI
st.markdown('<h1 class="main-header">📚 Multi-Modal RAG PDF ChatBot</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<h2 class="sub-header">📋 Instructions</h2>', unsafe_allow_html=True)
    st.markdown("""
    1. **Upload your PDF** using the file uploader
    2. **Wait for processing** - this may take a few minutes
    3. **Ask questions** about your document
    4. **Get intelligent answers** based on text, tables, and images
    """)
    
    st.markdown('<h2 class="sub-header">⚙️ Features</h2>', unsafe_allow_html=True)
    st.markdown("""
    - **Multi-modal RAG**: Processes text, tables, and images
    - **Advanced AI**: Uses Mistral AI and Groq models
    - **Smart Retrieval**: ChromaDB vector search
    - **Visual Understanding**: Pixtral for image analysis
    """)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<h2 class="sub-header">📁 Upload PDF</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None and not st.session_state.pdf_processed:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        texts, tables, images = process_pdf(tmp_file_path)
        
        if texts or tables or images:
            text_summaries, table_summaries, image_summaries = create_summaries(texts, tables, images)
            retriever = create_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries)
            
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.pdf_processed = True
                st.success("✅ PDF processed successfully! You can now ask questions.")
        
        os.unlink(tmp_file_path)
    
    if st.button("🔄 Reset Chat", help="Clear chat history and start over"):
        st.session_state.messages = []
        st.session_state.pdf_processed = False
        st.session_state.retriever = None
        st.rerun()

with col2:
    st.markdown('<h2 class="sub-header">💬 Chat with your PDF</h2>', unsafe_allow_html=True)
    
    if not st.session_state.pdf_processed:
        st.info("👈 Please upload a PDF file first to start chatting!")
    else:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask a question about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            chain = (
                {
                    "context": st.session_state.retriever | RunnableLambda(parse_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(build_prompt)
                | ChatMistralAI(model="mistral-large-latest")
                | StrOutputParser()
            )
            
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I encountered an error while processing your question."
                    })
            st.rerun()

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #111; font-size: 0.85rem;'>
        Multi-Modal RAG PDF ChatBot | Built with Streamlit, LangChain, and Mistral AI
    </div>
    """, 
    unsafe_allow_html=True
)