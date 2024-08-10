# Stock News Analysis and Prediction

## Project Overview
This project investigates the impact of news headlines on the price action of the S&P 500 stock index. Utilizing various sentiment analysis methods and machine learning models, we aim to predict price movements based on the sentiment derived from news headlines. This project involves data collection, preprocessing, sentiment analysis, knowledge graph construction, and LSTM-based model building.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Aims and Objectives](#project-aims-and-objectives)
- [Literature Review](#literature-review)
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Knowledge Graphs](#knowledge-graphs)
- [Model Building](#model-building)
- [Results and Discussion](#results-and-discussion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Project Aims and Objectives
### Research Aim
The aim of this research is to explore the effectiveness of sentiment analysis in predicting the price action of the S&P 500 stock index and to build a neural network model capable of accurately predicting the current closing stock price.

### Research Objectives
1. Build a dataset of news headlines and their respective previous and daily closing prices of the S&P 500 stock index.
2. Generate event tuples of the news headlines using knowledge graph methods.
3. Obtain sentiment scores for the event tuples using the Sentiment Intensity Analyzer and BERT.
4. Predict the S&P 500 stock index price using the sentiment scores and LSTM.

## Literature Review
The stock market is influenced by numerous factors, including market sentiment, which is significantly affected by news. Sentiment analysis has become a crucial tool for predicting stock prices, leveraging advancements in natural language processing and big data. This project compares various sentiment analysis methods and integrates them with machine learning models to predict stock prices accurately.

## Data Collection and Preprocessing
### Data Sources
We collected news headlines from three primary sources:
- CNBC
- Guardian
- Reuters

These datasets are stored in CSV files within the `data` directory.

### Data Preprocessing
The `StockNewsDataset.ipynb` notebook handles data preprocessing which includes:
- Loading datasets from CSV files.
- Handling missing values and splitting date-time information.
- Merging news headlines with corresponding S&P 500 stock prices.

### Key Code Snippets
```python
import pandas as pd

# Load datasets
dataCNBC = pd.read_csv('data/cnbc_headlines.csv')
datagaurd = pd.read_csv('data/guardian_headlines.csv')
datareuters = pd.read_csv('data/reuters_headlines.csv')

# Drop NA values and split time
dataCNBC = dataCNBC.dropna()
dataCNBC[['Timestamp', 'Time']] = dataCNBC['Time'].str.split(', ', expand=True)

# Merging datasets (example with CNBC data)
stock_data = pd.read_csv('data/sp500_prices.csv')
merged_data = pd.merge(dataCNBC, stock_data, left_on='Timestamp', right_on='Date')
merged_data = merged_data.drop(['Date'], axis=1)
```

## Knowledge Graphs
### Purpose
The purpose of constructing knowledge graphs is to extract and visualize the relationships between entities in the news headlines.

### Methodology
The `Knowledge_graphs.ipynb` notebook covers:
- Using NLP techniques to extract entities and relationships from the text.
- Building knowledge graphs to represent these relationships.

### Key Code Snippets
```python
import spacy
import pandas as pd

# Load Spacy model and data
nlp = spacy.load('en_core_web_sm')
df = pd.read_excel('data/cleaned_data.xlsx')

# Extract entities function
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Apply extraction to dataframe
df['entities'] = df['headline'].apply(extract_entities)

# Example entity extraction visualization
doc = nlp(df['headline'][0])
spacy.displacy.render(doc, style='ent', jupyter=True)
```

## Model Building
### Sentiment Intensity Analyzer
This section covers sentiment analysis using the Sentiment Intensity Analyzer and integrating sentiment scores into the dataset.

### Key Code Snippets
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

# Define a function to extract the sentiment scores
def get_sentiment_scores(text):
    scores = sid.polarity_scores(text)
    return pd.Series({
        'pos_score': scores['pos'],
        'neg_score': scores['neg'],
        'neu_score': scores['neu'],
        'compound_score': scores['compound']
    })

# Analyze sentiment for each entity and relationship
entity_sentiment_scores = df['entities'].apply(lambda x: get_sentiment_scores(x[0] if x else ""))
relationship_sentiment_scores = df['relationships'].apply(lambda x: get_sentiment_scores(x[0] if x else ""))

# Add sentiment scores back to the dataframe
df = pd.concat([df, entity_sentiment_scores.add_prefix('entity_')], axis=1)
df = pd.concat([df, relationship_sentiment_scores.add_prefix('relationship_')], axis=1)

# Selecting relevant columns for modeling
df = df[['Date', 'entity_pos_score', 'entity_neg_score', 'entity_neu_score', 'entity_compound_score',
         'relationship_pos_score', 'relationship_neg_score', 'relationship_neu_score', 'relationship_compound_score',
         'prevclose', 'Price']]

# Creating test set and cross-validation set
test_set = df.sample(n=5000, random_state=42)
df = df.drop(test_set.index)
crossdf = df
```

### LSTM Model with Sentiment Intensity Analyzer
### Key Code Snippets
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Split the data into training and testing sets
X = df.drop(['Date', 'Price'], axis=1)
y = df['Price']

# Create a separate variable for the indices
indices = df.index.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Training the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Predicting
predictions = model.predict(X_test)

# Visualization of results
plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```

### BERT-LSTM Model
### Key Code Snippets
```python
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the training set
max_len = 100
train_input = tokenizer(df['headline'].tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="tf")
test_input = tokenizer(test_set['headline'].tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="tf")

# Create a BERT-based model
input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert_model([input_ids, input_mask])[0]
X = tf.keras.layers.LSTM(50, return_sequences=True)(embeddings)
X = tf.keras.layers.LSTM(50)(X)
X = tf.keras.layers.Dense(1)(X)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=X)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='

mean_squared_error')

# Training the BERT-LSTM model
history = model.fit({'input_ids': train_input['input_ids'], 'attention_mask': train_input['attention_mask']}, 
                    df['Price'], 
                    epochs=10, 
                    batch_size=16, 
                    validation_split=0.2)

# Evaluate the model
loss = model.evaluate({'input_ids': test_input['input_ids'], 'attention_mask': test_input['attention_mask']}, 
                      test_set['Price'])
print(f'Test Loss: {loss}')

# Predicting
predictions = model.predict({'input_ids': test_input['input_ids'], 'attention_mask': test_input['attention_mask']})

# Visualization of results
plt.plot(test_set['Price'].values, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```

### Cross-Validation
### Key Code Snippets
```python
from sklearn.model_selection import KFold

# Perform cross-validation
kf = KFold(n_splits=5)
history = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    # Build and train model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                              validation_data=(X_val, y_val), callbacks=[es])
    history.append(model_history)

# Evaluate the model
scores = model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), y_test)
print('MSE: ', scores)

# Plot accuracy and validation loss for each epoch
train_losses = []
val_losses = []

for hist in history:
    train_losses.append(hist.history['loss'])
    val_losses.append(hist.history['val_loss'])

train_losses = np.mean(np.array(train_losses), axis=0)
val_losses = np.mean(np.array(val_losses), axis=0)

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('No of Epochs')
plt.ylabel('Loss')
plt.title('Evaluation of Cross-validation on BERT-LSTM')
plt.legend()
plt.show()
```

## Results and Discussion
### Results
- The models showed high accuracy in predicting S&P 500 prices.
- Cross-validation confirmed the models' effectiveness on unseen data.
- The BERT-LSTM model slightly outperformed the SentimentIntensityAnalyzer-LSTM model.

### Discussion
- The models effectively captured the sentiment from news headlines to predict stock prices.
- Some limitations include the use of a single financial instrument and the need for additional input variables.

## Future Work
- Use larger and more diverse datasets covering different timeframes and financial instruments.
- Explore other sentiment analysis models and neural network architectures.
- Incorporate additional variables like macroeconomic indicators for improved predictions.

## Acknowledgements
- **Supervisor**: Ms. Renuga Jayakumar
- **Module Leader**: Mr. Bente Riegler
- **Institution**: University of Hertfordshire

For more information refer to the project report. 
