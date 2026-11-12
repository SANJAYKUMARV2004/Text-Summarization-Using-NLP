"""Extractive and abstractive text summarization utilities."""
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def _ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

def extractive_summary(text: str, ratio: float = 0.3) -> str:
    """Rank sentences by normalized word frequency and return top sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("Please provide some text.")
    _ensure_nltk_data()
    sentences = sent_tokenize(cleaned)
    if len(sentences) <= 2:
        return cleaned

    stop_words = set(stopwords.words("english"))
    words = [
        word.lower() for word in word_tokenize(cleaned)
        if word.isalpha() and word.lower() not in stop_words
    ]
    frequencies = Counter(words)
    if not frequencies:
        return sentences[0]

    maximum = max(frequencies.values())
    for word in list(frequencies):
        frequencies[word] /= maximum

    scores = {}
    for index, sentence in enumerate(sentences):
        sentence_words = [
            word.lower() for word in word_tokenize(sentence)
            if word.isalpha() and word.lower() not in stop_words
        ]
        if sentence_words:
            scores[index] = sum(frequencies.get(word, 0) for word in sentence_words) / len(sentence_words)

    count = max(1, min(len(sentences), round(len(sentences) * ratio)))
    selected = sorted(sorted(scores, key=scores.get, reverse=True)[:count])
    return " ".join(sentences[index] for index in selected)

def abstractive_summary(text: str) -> str:
    """Generate a new summary using a pretrained Hugging Face transformer."""
    if not text.strip():
        raise ValueError("Please provide some text.")
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Chunk long input to stay within the model's input token limit.
    tokenizer = summarizer.tokenizer
    token_ids = tokenizer(text, truncation=False)["input_ids"]
    chunk_size = 900
    chunks = []
    for start in range(0, len(token_ids), chunk_size):
        chunk = tokenizer.decode(token_ids[start:start + chunk_size], skip_special_tokens=True)
        if chunk.strip():
            chunks.append(chunk)
    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(result[0]["summary_text"])
    combined = " ".join(summaries)
    if len(summaries) > 1:
        final = summarizer(combined, max_length=150, min_length=35, do_sample=False)
        return final[0]["summary_text"]
    return combined
