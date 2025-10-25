import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Dispute Notepad Assistant",
    page_icon="🧾",
    layout="wide"
)

st.title("🧾 Dispute Processing Notepad (Testing)")

# -----------------------------------
# LOAD VECTOR STORE
# -----------------------------------
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

# -----------------------------------
# LOAD LLM (GROQ LLAMA)
# -----------------------------------
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

# -----------------------------------
# BUILD QA CHAIN (if ready)
# -----------------------------------
if llm is not None and db is not None:
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=False
    )
else:
    qa = None

# -----------------------------------
# DISPUTE TEXT INPUT
# -----------------------------------
dispute_text = st.text_area("✍️ Paste dispute details below:", height=150)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("💡 Generate Recommendations")
with col2:
    st.caption("Uses Llama 3 model via Groq")

if analyze_btn:
    if dispute_text.strip():
        with st.spinner("Analyzing dispute details..."):
            if missing_instructions:
                st.error("❌ Missing 'Dispute instruction part 1.txt' file.")
            elif qa is None:
                st.error("⚠️ GROQ_API_KEY missing or model not available.")
            else:
                query = f"Given this dispute text, what should I do?\n\n{dispute_text}"
                answer = qa.run(query)
                st.subheader("🧠 Contextual Recommendations")
                st.write(answer)
    else:
        st.warning("Please enter some dispute details first.")

st.divider()
st.caption("Powered by Groq Llama 3 · LangChain · Streamlit")

# -----------------------------------
# 📋 DISPUTE INFORMATION FORM (COMPACT)
# -----------------------------------
st.subheader("📑 Dispute Information (Copy-Only)")

st.info("Fill in the details and click **Copy Form Data** to copy all fields together.", icon="🧾")

# --- Section 1: Basic details ---
col1, col2, col3 = st.columns(3)
with col1:
    dispute_id = st.text_input("Dispute ID", "1234")
    account = st.text_input("Account", "A-1234566")
    mpxn = st.text_input("MPXN", "1234567890123")
with col2:
    other_supplier = st.text_input("Other Supplier", "Spow")
    meter_number = st.text_input("Meter Number")
    ssd_date = st.date_input("SSD Date")
with col3:
    supply_status = st.selectbox("Supply Status", ["Loss", "Active", "Inactive", "Pending"])
    other_chl_comms = st.text_input("Other CHL Comms (Ref)")
    mmu_form = st.text_input("MMU/Settlement Form ID")

# --- Section 2: Reads (side by side) ---
st.markdown("#### 🔢 Reads")
col1, col2 = st.columns(2)

with col1:
    cos_read = st.text_input("COS Read")
    proposed_read = st.text_input("Proposed Read")

    st.markdown("**Back Up Read Set 1**")
    bu_r1_read = st.text_input("R1 Read (Set 1)")
    bu_r1_date = st.date_input("R1 Date (Set 1)")
    bu_r2_read = st.text_input("R2 Read (Set 1)")
    bu_r2_date = st.date_input("R2 Date (Set 1)")

with col2:
    st.markdown("**Back Up Read Set 2**")
    bu2_r1_read = st.text_input("R1 Read (Set 2)")
    bu2_r1_date = st.date_input("R1 Date (Set 2)")
    bu2_r2_read = st.text_input("R2 Read (Set 2)")
    bu2_r2_date = st.date_input("R2 Date (Set 2)")
    remark = st.text_area("Remarks / Dispute Summary", height=130)

# -----------------------------------
# COPYABLE OUTPUT
# -----------------------------------
form_summary = f"""
📑 **Dispute Information Summary**

**Dispute ID:** {dispute_id}
**Account:** {account}
**MPXN:** {mpxn}
**Other Supplier:** {other_supplier}
**Meter Number:** {meter_number}
**SSD Date:** {ssd_date}
**Supply Status:** {supply_status}

**COS Read:** {cos_read}
**Proposed Read:** {proposed_read}

**Back Up Read Set 1:**  
 R1: {bu_r1_read} ({bu_r1_date})  
 R2: {bu_r2_read} ({bu_r2_date})

**Back Up Read Set 2:**  
 R1: {bu2_r1_read} ({bu2_r1_date})  
 R2: {bu2_r2_read} ({bu2_r2_date})

**Remarks:**  
{remark}

**Other CHL Comms:** {other_chl_comms}  
**MMU/Settlement Form ID:** {mmu_form}
"""

st.markdown("### 🗒️ Preview")
st.text_area("Generated Summary", form_summary, height=200)

# Simple download/copy option
st.download_button(
    label="📋 Copy Form Data",
    data=form_summary,
    file_name="dispute_form.txt",
    mime="text/plain"
)
