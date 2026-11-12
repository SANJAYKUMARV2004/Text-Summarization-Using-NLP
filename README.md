# Text Summarization Using Natural Language Processing

A Streamlit application that shortens long text while retaining its important information. It provides two approaches: extractive summarization with NLTK and abstractive summarization with a pretrained Hugging Face transformer.

## Features
- Text input through a simple Streamlit interface
- Extractive summarization using sentence ranking and word frequency
- Abstractive summarization using DistilBART
- Adjustable extractive summary length
- Download generated summaries as text files

## Technologies
Python, NLTK, Hugging Face Transformers, PyTorch, Streamlit, Google Colab (optional for notebook-based experimentation).

## Setup
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501.

## Google Colab
You can also upload `app.py` and `summarizer.py` to Google Colab, install the dependencies with `!pip install -r requirements.txt`, and run Streamlit using a suitable tunnel if you want to expose the interface.

## How it works
1. The user enters text.
2. Extractive mode scores sentences using normalized word frequencies and selects the highest-scoring sentences in their original order.
3. Abstractive mode uses the pretrained `sshleifer/distilbart-cnn-12-6` model to generate a paraphrased summary. Long inputs are split into chunks.

## Notes
- The first abstractive run downloads model weights and may take time and disk space.
- The project currently summarizes English text.
- Machine-generated summaries can omit details or introduce inaccuracies. Review important summaries before relying on them.

## Author
SANJAYKUMAR V
