
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Download VADER lexicon
nltk.download('vader_lexicon')

# 1. Load Data
# df = pd.read_csv('reviews.csv')
data = [
    "I love this product! It's amazing.",
    "This is the worst experience ever.",
    "It's okay, not great but not terrible.",
    "Absolutely fantastic service."
]
df = pd.DataFrame(data, columns=['text'])

# 2. Analyze Sentiment
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return sia.polarity_scores(text)

df['scores'] = df['text'].apply(get_sentiment)
df['compound'] = df['scores'].apply(lambda x: x['compound'])
df['label'] = df['compound'].apply(lambda x: 'pos' if x > 0.05 else ('neg' if x < -0.05 else 'neu'))

print(df[['text', 'label', 'compound']])

# 3. Visualize
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Sentiment Distribution')
plt.show()
