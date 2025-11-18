import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Dispute Notepad Assistant", page_icon="🧾", layout="wide")

# --- GLOBAL STYLE TWEAKS (Compact Look) ---
st.markdown("""
    <style>
        .block-container {padding-top: 0.8rem; padding-bottom: 0.8rem; padding-left: 1.5rem; padding-right: 1.5rem;}
        .stTextInput, .stTextArea, .stSelectbox, .stDateInput label {font-size: 0.9rem !important;}
        textarea, input {padding: 0.3rem 0.4rem !important; font-size: 0.9rem !important;}
        h3, h4, h5, h6 {margin-bottom: 0.3rem;}
        .stButton>button {padding: 0.3rem 0.8rem; font-size: 0.9rem;}
        div[data-testid="column"] {padding: 0.2rem 0.5rem;}
    </style>
""", unsafe_allow_html=True)

# ---------------- PAGE HEADER ----------------
st.title("🧾 Dispute Processing Notepad (testing)")

# ---------------- EMBEDDING SETUP ----------------
@st.cache_resource
def load_vector_store():
    try:
        with open("Dispute instruction part 1.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
    except FileNotFoundError:
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts([text_data], embeddings)

db = load_vector_store()
missing_instructions = db is None

# ---------------- LLM SETUP ----------------
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

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever()) if (llm and db) else None

# ---------------- MAIN LAYOUT ----------------
left_col, right_col = st.columns([2.2, 1.5], gap="medium")

# LEFT PANEL -------------------
with left_col:
    st.markdown("#### ✍️ Dispute Details / Notepad")
    dispute_text = st.text_area(
        "Enter or paste dispute details below:",
        height=270,
        placeholder="Type or paste your dispute notes here..."
    )

    if st.button("💡 Generate Contextual Recommendations", use_container_width=True):
        if dispute_text.strip():
            with st.spinner("Analyzing dispute details using Llama 3..."):
                if missing_instructions:
                    st.error("❌ File 'Dispute instruction part 1.txt' not found.")
                elif qa is None:
                    st.error("❌ GROQ API key missing or LLM unavailable.")
                else:
                    query = f"Given this dispute text, what should I do?\n\n{dispute_text}"
                    answer = qa.run(query)
                    st.subheader("🧠 Recommendations")
                    st.write(answer)
        else:
            st.warning("Please enter some dispute details first.")

# RIGHT PANEL -------------------
with right_col:
    st.markdown("#### 📋 Dispute Data Form (Inline Compact View)")

    def inline_input(label, key=None, default=""):
        col1, col2 = st.columns([0.45, 0.55])
        with col1:
            st.markdown(f"<div style='text-align:right; padding-top:4px;'><b>{label}:</b></div>", unsafe_allow_html=True)
        with col2:
            return st.text_input("", value=default, key=key, label_visibility="collapsed")

    # Basic dispute fields
    dispute_id = inline_input("Dispute ID", "dispute_id", "1234")
    account = inline_input("Account", "account", "A-1234566")
    mpxn = inline_input("MPXN", "mpxn", "1234567890123")
    other_supplier = inline_input("Other Supplier", "supplier", "Spow")
    meter_number = inline_input("Meter Number", "meter_number")
    ssd_date = inline_input("SSD Date", "ssd_date")
    supply_status = inline_input("Supply Status", "supply_status", "Loss")
    cos_read = inline_input("COS Read", "cos_read")
    proposed_read = inline_input("Proposed Read", "proposed_read")

    st.markdown("##### 📦 Backup Read Set 1")
    r1_set1 = inline_input("R1 (Read & Date)", "r1s1")
    r2_set1 = inline_input("R2 (Read & Date)", "r2s1")

    st.markdown("##### 📦 Backup Read Set 2")
    r1_set2 = inline_input("R1 (Read & Date)", "r1s2")
    r2_set2 = inline_input("R2 (Read & Date)", "r2s2")

    remark = st.text_area("🗒 Remark / Summary", height=60)
    other_chl = inline_input("Other CHL Comms", "other_chl")
    mmu_form = inline_input("MMU/Settlement Form", "mmu_form")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📋 Copy Form Data", use_container_width=True):
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

# FOOTER
st.markdown("<hr><center><small>Powered by Groq Llama 3 · LangChain · Streamlit</small></center>", unsafe_allow_html=True)


