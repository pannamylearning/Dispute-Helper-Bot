import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

st.set_page_config(page_title="Dispute Notepad Assistant", page_icon="🧾", layout="centered")

st.title("🧾 Dispute Processing Notepad (Testing)")

# Build embeddings and FAISS vector store (cached to avoid rebuild each rerun)
@st.cache_resource
def load_vector_store():
    try:
        with open("Dispute instruction part 1.txt", "r", encoding="utf-8") as f:
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
    try:
        has_key = "GROQ_API_KEY" in st.secrets
    except StreamlitSecretNotFoundError:
        has_key = False
    if not has_key:
        return None
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model="llama-3.1-8b-instant"
    )

llm = load_llm()

# Build RetrievalQA if LLM and DB available
if llm is not None and db is not None:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
else:
    qa = None

# ------------------- MAIN DISPUTE TEXT AREA -------------------
dispute_text = st.text_area("✍️ Enter or paste dispute details:", height=200)

if st.button("💡 Generate Contextual Recommendations"):
    if dispute_text.strip():
        with st.spinner("Analyzing dispute details using Llama 3..."):
            if missing_instructions:
                st.error("❌ File 'Dispute instruction part 1.txt' not found in the repo. Add it to enable vector search and recommendations.")
            elif qa is None:
                st.error("GROQ API key not configured or LLM unavailable. Add `GROQ_API_KEY` to Streamlit secrets to enable the Llama 3 model.")
            else:
                query = f"Given this dispute text, what should I do?\n\n{dispute_text}"
                answer = qa.run(query)
                st.subheader("🧠 Recommendations")
                st.write(answer)
    else:
        st.warning("Please enter some dispute details first.")

# ------------------- DISPUTE DATA FORM -------------------
st.markdown("---")
st.header("📋 Dispute Data Form")

col1, col2 = st.columns(2)
with col1:
    dispute_id = st.text_input("Dispute ID", "1234")
    account = st.text_input("Account", "A-1234566")
    mpxn = st.text_input("MPXN", "1234567890123")
    other_supplier = st.text_input("Other Supplier", "Spow")
    meter_number = st.text_input("Meter Number")
with col2:
    ssd_date = st.date_input("SSD Date")
    supply_status = st.selectbox("Supply Status", ["Loss", "Active", "Inactive"])
    cos_read = st.text_input("COS Read")
    proposed_read = st.text_input("Proposed Read")

st.subheader("📦 Backup Read Set 1")
r1_set1 = st.text_input("R1 (Read & Date)")
r2_set1 = st.text_input("R2 (Read & Date)")

st.subheader("📦 Backup Read Set 2")
r1_set2 = st.text_input("R1 (Read & Date) ")
r2_set2 = st.text_input("R2 (Read & Date) ")

remark = st.text_area("🗒 Remark (Dispute Summary)")
other_chl = st.text_input("Other CHL Comms (Ref Field)")
mmu_form = st.text_input("MMU/Settlement Form")

# ------------------- COPY BUTTON -------------------
if st.button("📋 Copy Form Data"):
    combined_data = f"""
Dispute ID: {dispute_id}
Account: {account}
MPXN: {mpxn}
Other Supplier: {other_supplier}
Meter Number: {meter_number}
SSD Date: {ssd_date}
Supply Status: {supply_status}

COS Read: {cos_read}
Proposed Read: {proposed_read}

Backup Read Set 1:
  R1: {r1_set1}
  R2: {r2_set1}

Backup Read Set 2:
  R1: {r1_set2}
  R2: {r2_set2}

Remark: {remark}

Other CHL Comms: {other_chl}
MMU/Settlement Form: {mmu_form}
"""
    st.code(combined_data, language="markdown")
    st.info("✅ Copy the above formatted data manually for your record or processing.")

st.markdown("---")
st.caption("Powered by Groq Llama 3 · LangChain · Streamlit")
