import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="Dispute Notepad Assistant", page_icon="🧾", layout="centered")

st.title("🧾 Dispute Processing Notepad (Free Model)")

# Build embeddings and FAISS vector store (cached to avoid rebuild each rerun)
@st.cache_resource
def load_vector_store():
    # Read the instruction file at runtime so missing files don't cause
    # a NameError at import time. Return None if file is missing.
    try:
        with open("dispute_work_instructions.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts([text_data], embeddings)
    return db

db = load_vector_store()
missing_instructions = db is None

# Initialize free Groq model (Llama 3)
@st.cache_resource
def load_llm():
    # Return None if the required secret isn't provided so the app
    # can start safely in dev environments. The UI will show a helpful
    # message instead of crashing with an ImportError/KeyError.
    try:
        has_key = "GROQ_API_KEY" in st.secrets
    except StreamlitSecretNotFoundError:
        has_key = False
    if not has_key:
        return None
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama3-8b-8192"
    )

llm = load_llm()

# Only build the RetrievalQA chain when an LLM is available.
if llm is not None and db is not None:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
else:
    qa = None

# User interface
dispute_text = st.text_area("✍️ Enter or paste dispute details:", height=200)

if st.button("💡 Generate Contextual Recommendations"):
    if dispute_text.strip():
        with st.spinner("Analyzing dispute details using Llama 3..."):
            if missing_instructions:
                st.error("❌ File 'Dispute instruction part.txt' not found in the repo. Add it to enable vector search and recommendations.")
            elif qa is None:
                # If QA is None here, either the GROQ API key is missing or
                # some other runtime dependency failed to initialize.
                st.error("GROQ API key not configured or LLM unavailable. Add `GROQ_API_KEY` to Streamlit secrets to enable the Llama 3 model.")
            else:
                query = f"Given this dispute text, what should I do?\n\n{dispute_text}"
                answer = qa.run(query)
                st.subheader("🧠 Recommendations")
                st.write(answer)
    else:
        st.warning("Please enter some dispute details first.")

st.markdown("---")
st.caption("Powered by Groq Llama 3 · LangChain · Streamlit")


