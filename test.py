import streamlit as st
import re
import pandas as pd
import plotly.express as px
from rouge_score import rouge_scorer
from transformers import pipeline

# Placeholder user database
USER_DATABASE = {
    "uzair1": "12345678",
    "uzair2": "12345678",
    "uzair3": "12345678"
}

# Function to preprocess the text
def preprocess(text):
    text is re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Function to summarize text using transformers pipeline
def summarize_text(text, min_length=30, max_length=150):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Function to calculate ROUGE scores
def calculate_rouge_scores(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores

# Plot summary metrics
def plot_summary_metrics(article, summary, rouge_scores):
    metrics = {
        "Length (words)": [len(article.split()), len(summary.split())],
        "Sentences": [article.count('.'), summary.count('.')],
        "ROUGE-1 Precision": [rouge_scores['rouge1'].precision],
        "ROUGE-1 Recall": [rouge_scores['rouge1'].recall],
        "ROUGE-1 F1": [rouge_scores['rouge1'].fmeasure],
        "ROUGE-2 Precision": [rouge_scores['rouge2'].precision],
        "ROUGE-2 Recall": [rouge_scores['rouge2'].recall],
        "ROUGE-2 F1": [rouge_scores['rouge2'].fmeasure],
        "ROUGE-L Precision": [rouge_scores['rougeL'].precision],
        "ROUGE-L Recall": [rouge_scores['rougeL'].recall],
        "ROUGE-L F1": [rouge_scores['rougeL'].fmeasure]
    }
    df = pd.DataFrame(metrics, index=["Article", "Summary"])
    fig = px.bar(df, barmode='group', title="Summary Metrics")
    st.plotly_chart(fig)

# Main function to run the Streamlit app
def main():
    st.title("PubMed Article Summarizer")
    st.write("Upload a PubMed article to get a summary.")
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # User authentication
    if not st.session_state.authenticated:
        user = st.text_input("Enter your username:")
        password = st.text_input("Enter your password:", type="password")
        login_button = st.button("Enter")
        
        if login_button:
            if user in USER_DATABASE and USER_DATABASE[user] == password:
                st.session_state.authenticated = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    
    if st.session_state.authenticated:
        # Summary length/style options
        min_length = st.slider("Minimum summary length (words):", 30, 100, 30)
        max_length = st.slider("Maximum summary length (words):", 100, 300, 150)

        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type="txt")

        if uploaded_file is not None:
            try:
                st.write("File successfully uploaded.")
                article_text = uploaded_file.read().decode("utf-8")
                st.write("File read and decoded.")
                article_text = preprocess(article_text)
                st.write("Text preprocessed.")
                st.write("Original Article:")
                st.write(article_text)
                
                summary = summarize_text(article_text, min_length, max_length)
                st.write("Text summarized.")
                st.write("AI Summary:")
                st.write(summary)
                
                if "Error during summarization" not in summary:
                    rouge_scores = calculate_rouge_scores(article_text, summary)
                    st.write("ROUGE scores calculated.")
                    plot_summary_metrics(article_text, summary, rouge_scores)
                    st.write("Metrics plotted.")
                else:
                    st.error(summary)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

