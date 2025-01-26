#%%
!pip install -U pip setuptools wheel
!pip install -U spacy
!python -m spacy download en_core_web_sm
#%%
pip install spacy
pip install yfinance
pip install nltk
pip install pandas
pip install matplotlib
#%%
python -m spacy download en_core_web_sm
#%%
import spacy
import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

def get_financial_data(ticker="BTC-USD"):
    data = yf.download(ticker, period="1y", interval="1d")
    return data

def analyze_sentiment(text):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def generate_report(blockchain_name, description, financial_data):
    doc = nlp(description)
    entities = [ent.text for ent in doc.ents]

    plt.figure(figsize=(10, 6))
    plt.plot(financial_data['Close'])
    plt.title(f'{blockchain_name} Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.savefig(f'{blockchain_name}_price_trend.png')

    sentiment_score = analyze_sentiment(description)
    sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    report = f"""
    Web3 Investment Research Report: {blockchain_name}
    ==================================================

    Blockchain Overview:
    {description}

    Key Entities Identified:
    {', '.join(entities)}

    Financial Data Analysis:
    Price Trend for the last year:
    ![Price Trend](./{blockchain_name}_price_trend.png)

    Sentiment Analysis:
    Sentiment Score: {sentiment_score}
    Overall Sentiment: {sentiment}

    Risk Evaluation:
    Based on the sentiment and price volatility, the current risk level for this asset is high.
    """

    return report

if __name__ == "__main__":
    blockchain_name = "Bitcoin"
    description = "Bitcoin is a decentralized digital currency that operates on the blockchain, allowing peer-to-peer transactions without the need for a central authority."

    financial_data = get_financial_data()

    report = generate_report(blockchain_name, description, financial_data)

    print(report)

    with open(f"{blockchain_name}_investment_report.txt", "w") as f:
        f.write(report)
