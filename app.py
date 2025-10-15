import streamlit as st

st.set_page_config(page_title="Dispute Notepad", layout="wide")

st.title("?? Dispute Processing Notepad")
st.write("Your AI helper for handling dispute accounts efficiently.")

# User input area
user_input = st.text_area("Enter dispute details or notes here:", height=200)

# Display contextual action suggestions (will be dynamic later)
if user_input:
    st.write("### ?? Suggested Next Actions:")
    st.markdown("- Check backup reads for rate calculation")
    st.markdown("- Verify supplier cost rate difference")
    st.markdown("- Escalate if data mismatch persists")
