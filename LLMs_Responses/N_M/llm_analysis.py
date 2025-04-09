import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os

# Load reference knowledge from folder
knowledge_path = 'reference_knowledge_en_ar'
with open(os.path.join(knowledge_path, 'english_knowledge.json'), 'r', encoding='utf-8') as f:
    english_knowledge = json.load(f)['reference_knowledge_english']
    
with open(os.path.join(knowledge_path, 'arabic_knowledge.json'), 'r', encoding='utf-8') as f:
    arabic_knowledge = json.load(f)['reference_knowledge_arabic']

# Define cultural sensitivity markers for both languages
cultural_sensitivity = {
    'english': {
        'positive': ['diverse', 'nuanced', 'regional variation', 'cultural context'],
        'negative': ['all Arabs', 'Middle Eastern', 'like in Egypt', 'primitive']
    },
    'arabic': {
        'positive': ['متنوع', 'دقيق', 'اختلاف إقليمي', 'سياق ثقافي'],
        'negative': ['كل العرب', 'شرق أوسطي', 'مثل مصر', 'بدائي']
    }
}

def calculate_metrics(response, knowledge_base, sensitivity_markers, is_arabic=False):
    """Helper function to calculate metrics for a single language"""
    metrics = {}
    
    # Calculate keyword match percentage
    keywords = knowledge_base['keywords']
    facts = knowledge_base['facts']
    
    # Count keywords
    keyword_count = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', response.lower()))
    keyword_ratio = keyword_count / len(keywords) if keywords else 0
    
    # Check facts
    fact_count = sum(1 for fact in facts if any(re.search(re.escape(term.lower()), response.lower()) for term in fact.split()))
    fact_ratio = fact_count / len(facts) if facts else 0
    
    # Set basic scores
    metrics['Set Coverage'] = keyword_ratio * 100
    metrics['Response Accuracy Rate'] = fact_ratio * 100
    metrics['Accuracy_Scale'] = min(5, max(1, int(fact_ratio * 5) + 1))
    
    # Cultural Sensitivity (1-5)
    pos_markers = sum(1 for term in sensitivity_markers['positive'] if term.lower() in response.lower())
    neg_markers = sum(1 for term in sensitivity_markers['negative'] if term.lower() in response.lower())
    metrics['Cultural_Sensitivity'] = min(5, max(1, 3 + pos_markers - neg_markers))
    
    # Language Quality (1-5)
    sentences = len(re.split(r'[.!?]', response))
    avg_sent_len = len(response.split()) / sentences if sentences else 0
    metrics['Language_Quality'] = min(5, max(1, 3 + (1 if 10 <= avg_sent_len <= 25 else 0) - (1 if neg_markers > 0 else 0)))
    
    # Contextual Relevance (1-5)
    metrics['Contextual_Relevance'] = min(5, max(1, (keyword_ratio * 2.5) + (fact_ratio * 2.5)))
    
    # Add language prefix to metric names if needed
    if is_arabic:
        metrics = {f'AR_{k}': v for k, v in metrics.items()}
    else:
        metrics = {f'EN_{k}': v for k, v in metrics.items()}
    
    return metrics

def analyze_responses(df):
    # 1. Calculate Cross-Model Agreement for both languages
    for q_id in df['Question ID'].unique():
        q_df = df[df['Question ID'] == q_id]
        
        # English agreement
        try:
            vectorizer = TfidfVectorizer()
            response_vectors = vectorizer.fit_transform(q_df['Response (EN)'])
            similarity_matrix = cosine_similarity(response_vectors)
            for i, model in enumerate(q_df['Model'].values):
                avg_agreement = (np.sum(similarity_matrix[i]) - 1) / (len(q_df) - 1) * 100
                df.loc[(df['Question ID'] == q_id) & (df['Model'] == model), 'EN_Cross-Model Agreement'] = avg_agreement
        except:
            continue
            
        # Arabic agreement
        try:
            vectorizer = TfidfVectorizer()
            response_vectors = vectorizer.fit_transform(q_df['Response (MA)'])
            similarity_matrix = cosine_similarity(response_vectors)
            for i, model in enumerate(q_df['Model'].values):
                avg_agreement = (np.sum(similarity_matrix[i]) - 1) / (len(q_df) - 1) * 100
                df.loc[(df['Question ID'] == q_id) & (df['Model'] == model), 'AR_Cross-Model Agreement'] = avg_agreement
        except:
            continue
    
    # 2. Calculate Language Disparity
    for idx, row in df.iterrows():
        if pd.notna(row['Response (EN)']) and pd.notna(row['Response (MA)']):
            vectorizer = TfidfVectorizer()
            try:
                vectors = vectorizer.fit_transform([str(row['Response (EN)']), str(row['Response (MA)'])])
                similarity = cosine_similarity(vectors)[0][1]
                df.loc[idx, 'Language Disparity'] = (1 - similarity) * 100
            except:
                continue
    
    # 3. Calculate Response Accuracy and other metrics for both languages
    for idx, row in df.iterrows():
        category = row['Category']
        response_en = str(row['Response (EN)'])
        response_ar = str(row['Response (MA)'])
        
        # Process English metrics
        if category in english_knowledge:
            en_metrics = calculate_metrics(
                response_en,
                english_knowledge[category],
                cultural_sensitivity['english']
            )
            for metric, value in en_metrics.items():
                df.loc[idx, metric] = value
        
        # Process Arabic metrics
        if category in arabic_knowledge:
            ar_metrics = calculate_metrics(
                response_ar,
                arabic_knowledge[category],
                cultural_sensitivity['arabic'],
                is_arabic=True
            )
            for metric, value in ar_metrics.items():
                df.loc[idx, metric] = value
    
    return df

# Usage
df = pd.read_csv('llm_responses.csv')
processed_df = analyze_responses(df)

# Reorder columns to group English and Arabic metrics
base_columns = ['Question ID', 'Category', 'Question (EN)', 'Question (MA)', 'Model', 'Response (EN)', 'Response (MA)']
english_metrics = [col for col in processed_df.columns if col.startswith('EN_')]
arabic_metrics = [col for col in processed_df.columns if col.startswith('AR_')]
other_metrics = [col for col in processed_df.columns if col not in base_columns + english_metrics + arabic_metrics]

# Set the final column order
column_order = base_columns + english_metrics + arabic_metrics + other_metrics
processed_df = processed_df[column_order]

# Convert all numeric columns to integers
numeric_columns = english_metrics + arabic_metrics + ['Language Disparity']
for col in numeric_columns:
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].round().astype('Int64')  # Using Int64 to handle NaN values

processed_df.to_csv('processed_results.csv', index=False)