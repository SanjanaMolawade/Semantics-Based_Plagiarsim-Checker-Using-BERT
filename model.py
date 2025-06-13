import pandas as pd
from tqdm import tqdm
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect, DetectorFactory
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the seed for language detection
DetectorFactory.seed = 0

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Preprocessing function to load data and get a sample
def preprocess_data(data_path, sample_size):
    data = pd.read_csv(data_path, low_memory=False)
    data = data.dropna(subset=['abstract']).reset_index(drop=True)
    data = data.sample(sample_size)[['abstract', 'cord_uid']]
    return data

# Function to create vectors from text using the BERT model
def create_vector_from_text(tokenizer, model, text, MAX_LEN=510):
    input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=MAX_LEN)
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    input_ids = results[0]
    attention_mask = [int(i > 0) for i in input_ids]
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        logits, encoded_layers = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
    
    vector = encoded_layers[12][0][0].detach().cpu().numpy()
    return vector

# Function to create a vector database from a set of abstracts
def create_vector_database(data):
    vectors = []
    source_data = data.abstract.values
    for text in tqdm(source_data):
        vector = create_vector_from_text(tokenizer, model, text)
        vectors.append(vector)
    
    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb).reshape(1, -1))
    return data

# Translation function
def translate_text(text, text_lang, target_lang='en'):
    model_name = f"Helsinki-NLP/opus-mt-{text_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    formatted_text = f">>{text_lang}<< {text}"
    translation = model.generate(**tokenizer([formatted_text], return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translation][0]
    return translated_text

# Process a document to generate vectors
def process_document(text):
    text_vect = create_vector_from_text(tokenizer, model, text)
    text_vect = np.array(text_vect).reshape(1, -1)
    return text_vect

# Check if similarity score indicates plagiarism
def is_plagiarism(similarity_score, plagiarism_threshold):
    return similarity_score >= plagiarism_threshold

# Candidate Languages for Translation
language_list = ['de', 'fr', 'el', 'ja', 'ru']

# Detect and translate the document if necessary
def check_incoming_document(incoming_document):
    text_lang = detect(incoming_document)
    if text_lang == 'en':
        return incoming_document
    elif text_lang not in language_list:
        return None
    else:
        return translate_text(incoming_document, text_lang)

# Run plagiarism analysis on a query text
def run_plagiarism_analysis(query_text, data, plagiarism_threshold=0.8):
    top_N = 3
    document_translation = check_incoming_document(query_text)
    
    if document_translation is None:
        return {
            'error': "Only the following languages are supported: English, French, Russian, German, Greek, and Japanese"
        }
    
    # Process the translated document to create its vector
    query_vect = process_document(document_translation)
    
    # Calculate cosine similarity with the vector database
    data["similarity"] = data["vectors"].apply(lambda x: cosine_similarity(query_vect, x)[0][0])
    similar_articles = data.sort_values(by='similarity', ascending=False).head(top_N + 1)

    # Calculate originality percentage and determine if matches are found
    similarity_score = similar_articles.iloc[0]["similarity"]
    originality_percentage = 100 - (similarity_score * 100)
    matches_found = any(similarity_score >= plagiarism_threshold for similarity_score in similar_articles['similarity'])
    
    most_similar_article = similar_articles.iloc[0]["abstract"]
    is_plagiarised = is_plagiarism(similarity_score, plagiarism_threshold)
    
    # Construct the plagiarism report
    plagiarism_decision = {
        'similarity_score': similarity_score,
        'originality_percentage': originality_percentage,
        'is_plagiarised': is_plagiarised,
        'matches_found': matches_found,
        'most_similar_article': most_similar_article,
        'article_submitted': query_text
    }
    
    return plagiarism_decision
