import streamlit as st
from summarizer import extractive_summary, abstractive_summary

st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")
st.title("📝 Text Summarization Using NLP")
st.write("Generate concise summaries from long text using extractive or abstractive NLP.")

text = st.text_area("Enter or paste your text", height=260, placeholder="Paste an article or document here...")
method = st.radio("Choose summarization method", ["Extractive (NLTK)", "Abstractive (Hugging Face)"], horizontal=True)
ratio = st.slider("Summary length (extractive)", min_value=0.1, max_value=0.6, value=0.3, step=0.05)

if st.button("Generate Summary", type="primary", use_container_width=True):
    if len(text.split()) < 30:
        st.warning("Please enter at least 30 words for a more useful summary.")
    else:
        with st.spinner("Generating your summary..."):
            try:
                if method.startswith("Extractive"):
                    summary = extractive_summary(text, ratio=ratio)
                else:
                    summary = abstractive_summary(text)
                st.subheader("Generated Summary")
                st.write(summary)
                st.download_button("Download summary", summary, file_name="summary.txt", mime="text/plain")
                st.caption(f"Original words: {len(text.split())} | Summary words: {len(summary.split())}")
            except Exception as exc:
                st.error(f"Could not generate summary: {exc}")

with st.expander("About this project"):
    st.write("Extractive summarization selects important sentences from the original text. Abstractive summarization uses a pretrained transformer model to generate new wording.")
