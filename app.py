import streamlit as st
from datetime import datetime

st.title("🧾 Smart Dispute Processing Notepad")

# Load work instruction (sample placeholder)
WORK_INSTRUCTIONS = {
    "backup read": "Check both backup reads and average them to calculate the Change of Supplier (COS) read.",
    "start date": "Validate the start date with MDM and confirm supplier switch date alignment.",
    "status check": "If account is inactive, validate the status in CRM before processing dispute.",
    "credit check": "Confirm if the credit block is lifted before re-processing the invoice.",
    "escalation": "Escalate only if SLA exceeded 48 hours or if variance > 20%."
}

# Text area for dispute input
dispute_text = st.text_area("Enter Dispute Details:", height=200)

if st.button("💡 Generate Recommendations"):
    st.markdown("### 🧠 Contextual Recommendations")
    found = False
    for key, tip in WORK_INSTRUCTIONS.items():
        if key in dispute_text.lower():
            st.success(f"🔹 *{key.title()}*: {tip}")
            found = True
    if not found:
        st.info("No specific recommendation found. Try adding more case details or keywords.")

