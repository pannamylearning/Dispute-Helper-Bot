import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="Dispute Notepad Assistant", page_icon="🧾", layout="centered")

st.title("🧾 Smart Dispute Processing Notepad (Free Model)")

# Load dispute work instructions
try:
    with open("dispute_work_instructions.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
except FileNotFoundError:
    st.error("❌ File 'dispute_work_instructions.txt' not found in repo.")
    st.stop()

# Build embeddings and FAISS vector store (cached to avoid rebuild each rerun)
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts([text_data], embeddings)
    return db

db = load_vector_store()

# Initialize free Groq model (Llama 3)
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama3-8b-8192"
    )

llm = load_llm()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=False
)

# User interface
dispute_text = st.text_area("✍️ Enter or paste dispute details:", height=200)

if st.button("💡 Generate Contextual Recommendations"):
    if dispute_text.strip():
        with st.spinner("Analyzing dispute details using Llama 3..."):
            query = f"Given this dispute text, what should I do?\n\n{dispute_text}"
            answer = qa.run(query)
        st.subheader("🧠 Recommendations")
        st.write(answer)
    else:
        st.warning("Please enter some dispute details first.")

st.markdown("---")
st.caption("Powered by Groq Llama 3 · LangChain · Streamlit")
