import streamlit as st

st.set_page_config(page_title="Dispute Notepad", layout="wide")

st.title("🧾 Dispute Processing Notepad")
st.write("Your AI helper for handling dispute accounts efficiently.")

# User input area
user_input = st.text_area("Enter dispute details or notes here:", height=200)

# Add a button aligned to the right below the text area
col1, col2, col3 = st.columns([6, 2, 1])  # adjust ratios for layout
with col3:
    generate = st.button("🚀 Generate Suggestions")

# Show suggestions only when button is clicked and text is entered
if generate and user_input:
    st.write("### 💡 Suggested Next Actions:")
    st.markdown("- Check backup reads for rate calculation")
    st.markdown("- Verify supplier cost rate difference")
    st.markdown("- Escalate if data mismatch persists")

elif generate and not user_input:
    st.warning("⚠️ Please enter dispute details before generating suggestions.")
