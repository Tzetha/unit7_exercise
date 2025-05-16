import numpy as np
import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from nltk import word_tokenize, FreqDist, ConditionalFreqDist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import math

# Download corpus
nltk.download('brown')
nltk.download('punkt')

# Load Brown news corpus
sentences = brown.sents(categories='news')
data = [' '.join(sent) for sent in sentences]

# ------------------------------------
# Trigram Model
# ------------------------------------
print("🔧 Building Trigram Model...")

# Preprocess text and collect trigrams
trigrams = []
for sentence in data:
    tokens = ['<s>'] + word_tokenize(sentence.lower()) + ['</s>']
    trigrams.extend(ngrams(tokens, 3))

# Build trigram frequency model
trigram_model = ConditionalFreqDist()
for w1, w2, w3 in trigrams:
    trigram_model[(w1, w2)][w3] += 1

# Trigram Prediction
def predict_with_trigram(seed_text, n_words=5):
    tokens = word_tokenize(seed_text.lower())
    if len(tokens) < 2:
        tokens = ['<s>'] + tokens
    output = tokens[:]
    for _ in range(n_words):
        context = tuple(output[-2:])
        if context in trigram_model:
            next_word = trigram_model[context].max()
        else:
            next_word = '</s>'
        output.append(next_word)
        if next_word == '</s>':
            break
    return ' '.join(output).replace('<s> ', '').replace(' </s>', '')

# Trigram Accuracy Evaluation
def evaluate_trigram_accuracy(trigram_model, data):
    correct = 0
    total = 0
    for sentence in data:
        tokens = ['<s>'] + word_tokenize(sentence.lower()) + ['</s>']
        for i in range(len(tokens) - 2):
            context = (tokens[i], tokens[i + 1])
            actual_next = tokens[i + 2]
            if context not in trigram_model:
                continue
            predicted = trigram_model[context].max()
            if predicted == actual_next:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# ------------------------------------
# LSTM Model
# ------------------------------------
print("🔧 Building LSTM Model...")

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i + 1]
        input_sequences.append(n_gram_seq)

# Custom padding
def custom_pad_sequences(sequences, maxlen=None, padding='pre', value=0):
    if not maxlen:
        maxlen = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < maxlen:
            pad_len = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * pad_len + seq
            else:
                seq = seq + [value] * pad_len
        else:
            seq = seq[-maxlen:]
        padded.append(seq)
    return np.array(padded)

max_seq_len = max(len(x) for x in input_sequences)
input_sequences = custom_pad_sequences(input_sequences, maxlen=max_seq_len)

# Split predictors and labels
input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# Define and train LSTM model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_seq_len - 1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X, y, epochs=10, batch_size=128, verbose=1)

# Evaluate LSTM model
eval_results = model.evaluate(X, y, verbose=0)

# LSTM Prediction
def predict_with_lstm(seed_text, n_words=5):
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = custom_pad_sequences([token_list], maxlen=max_seq_len - 1)
        predicted_index = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                seed_text += ' ' + word
                break
    return seed_text

# ------------------------------------
# Results
# ------------------------------------
print("\n📊 Evaluation Results:")
trigram_acc = evaluate_trigram_accuracy(trigram_model, data)
print(f"Trigram Accuracy (Top-1): {trigram_acc:.4f}")
print(f"LSTM Accuracy (Top-1): {eval_results[1]:.4f}")
print(f"LSTM Perplexity: {math.exp(eval_results[0]):.4f}")

seed = "The president said"
print("\n🔤 Trigram Prediction:")
print(predict_with_trigram(seed, 5))

print("\n🤖 LSTM Prediction:")
print(predict_with_lstm(seed, 5))
