from pathlib import Path
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
import spacy_curated_transformers

custom_stopwords_eng = [
    "a", "an", "the", "and", "or", "but", "for", "with", "in", "on", "at", "to", "from", "of", "by", "as", "is", "are",
    "were", "was", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "must", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "/", ",", ".", ":", ";", "(", ")",
    "[", "]", "{", "}", "'", "\"", "‘", "’", "“", "”", "he", "she", "it", "they", "we", "you", "that", "which", "who",
    "whom", "whose", "of", "to", "in", "for", "on", "with", "by", "at", "from", "into", "onto", "upon", "among",
    "between", "within", "without", "under", "over", "through", "during", "before", "after", "since", "about",
    "around", "against", "beside", "beyond", "above", "below", "throughout", "toward", "across", "is", "am", "are",
    "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "can", "could", "must", "good", "bad", "well", "better", "best", "often", "sometimes", "always",
    "never", "many", "few", "much", "little", "more", "less", "most", "least", "thou", "thee", "thine", "thy", "ye",
    "hath", "hast", "art", "didst", "doth", "wast", "were", "whence", "whom", "wilt"
]

nlp = spacy.load("en_core_web_trf")
print(spacy.info('en_core_web_trf'))

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tagged_tokens = nltk.pos_tag(tokens)

    english_stopwords = set(stopwords.words('english'))
    custom_stopwords_set = set(custom_stopwords_eng)
    all_stopwords_set = custom_stopwords_set.union(english_stopwords)

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in tagged_tokens if
              token.isalpha() and token not in all_stopwords_set]

    return tokens


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun (for consistency)


def perform_topic_modeling(text, num_topics=5):
    preprocessed_tokens = preprocess_text(text)
    # Join tokens back into a single string
    ntext = ' '.join(preprocessed_tokens)

    # Vectorize the text
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=.2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform([ntext])  # Ensure ntext is a string here

    # Apply NMF
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_model.fit(tfidf)

    # Extract the topics
    topics = []
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_words_per_topic = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-10 - 1:-1]  # Indices of top words for this topic
        top_features = [feature_names[i] for i in top_features_ind]  # Top words
        weights = [topic[i] for i in top_features_ind]  # Weights of the top words

        # Prepare top words and their weights for visualization
        top_words_info = [(word, weight) for word, weight in zip(top_features, weights)]
        top_words_per_topic.append(top_words_info)

        # For textual representation, if needed
        topic_str = " + ".join([f"{weight:.3f}*{word}" for word, weight in zip(top_features, weights)])
        topics.append((topic_idx, topic_str))

    return nmf_model.components_, top_words_per_topic


def generate_dependency_chart(text_input):
    doc = nlp(text_input)
    sentspans = list(doc.sents)
    options = {'color': '#ffffff', 'bg': '#2E86C1', 'compact': True}
    html = displacy.render(sentspans, style="dep", options=options, page=True)

    output_path = Path('dependency_chart.html')
    output_path.open("w", encoding="utf-8").write(html)
