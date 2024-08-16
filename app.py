import os
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from gensim.models import Word2Vec
import numpy as np


os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def summarize_text(content, summary_length=1):
    sentences = nltk.sent_tokenize(content)
    corpus = []
    lemmatize = WordNetLemmatizer()
    for sentence in sentences:
        words = word_tokenize(sentence)  
        filtered_words = []
        for word in words:
            word = re.sub(r"[,.()]", " ", word)
            word = re.sub("[^a-zA-Z]", "", word)
            word = lemmatize.lemmatize(word)
            word = word.lower()
            if word and word not in stopwords.words('english'):
                filtered_words.append(word)
        if filtered_words:
            corpus.append(filtered_words)


    model = Word2Vec(corpus, vector_size=100, window=5, min_count=1, workers=4)


    sv = []
    for sentence in corpus:
        wv = [model.wv[word] for word in sentence if word in model.wv]
        if wv:
            sentence_vector = np.mean(wv, axis=0)
        else:
            sentence_vector = np.zeros(model.vector_size)
        sv.append(sentence_vector)


    dv = np.mean(sv, axis=0)

 
    sentence_score = []
    for i, sentence_vector in enumerate(sv):
        score = np.dot(sentence_vector, dv)
        reconstructed_sentence = " ".join(corpus[i])
        sentence_score.append((reconstructed_sentence, score))

    sentence_score.sort(key=lambda x: x[1], reverse=True)
    summary = " ".join([sentence for sentence, score in sentence_score[:summary_length]])

    return summary


st.title("Text Summarization Tool")

st.write("Enter the text you want to summarize:")


text_input = st.text_area("Input Text", height=300)

summary_length = st.slider("Summary Length", min_value=1, max_value=5, value=1)

if st.button("Generate Summary"):
    if text_input:
        summary = summarize_text(text_input, summary_length)
        st.write("Summary:")
        st.write(summary)
    else:
        st.write("Please enter some text.")
