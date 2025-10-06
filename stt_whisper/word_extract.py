# pip install nltk pandas
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords

from collections import Counter
import string

nltk.download('punkt')
nltk.download('punkt_tab')   
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def extract_keywords(prompt, max_words=8):
    """
    Testing
    [x] more flexible keywords
    [x] allow shorter answers
    """
    # Tokenize and tag
    words = nltk.word_tokenize(prompt.lower())
    tagged = nltk.pos_tag(words)

    # Filter valid words
    words_filtered = [w for w in words
        if w.isalpha() and w not in stop_words and w not in string.punctuation]
    freq = Counter(words_filtered)

    pos_weights = {
        "NN": 3, "NNS": 3, "NNP": 3, "NNPS": 3,   # Nouns (subjects, objects)
        "VB": 2, "VBD": 2, "VBG": 2, "VBN": 2, "VBP": 2, "VBZ": 2,  # Verbs
        "JJ": 1.5, "JJR": 1.5, "JJS": 1.5,         # Adjectives
        "RB": 1, "RBR": 1, "RBS": 1                # Adverbs
    }
    scored = {}
    for word, pos in tagged:
        if word in freq and pos in pos_weights:
            scored[word] = freq[word] * pos_weights[pos]


    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    keywords = [w for w, s in ranked[:max_words]]
    if len(keywords) < 3: #Ensures at least 3 words. Change if too many none
        keywords += [w for w in words_filtered if w not in keywords][:3 - len(keywords)]

    return keywords


def find_prompt_with_words(words, df, min_matches=1):
    for prompt in df["long_prompt"].dropna():
        count = sum(1 for w in words if w.lower() in prompt.lower())
        if count >= min_matches:
            return prompt
    return None

