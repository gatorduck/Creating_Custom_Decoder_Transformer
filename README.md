# About

My motivation is inspired by current large language models trained on sequences of tokens representing natural language. In this case, each diagnosis code (like an ICD-10 code) is treated as a token in a sequence, just like a word in a sentence. But instead of natural language, the sequence represents a patient’s chronological medical history. The goal of this custom decoder is to learn temporal patterns or predict future diagnoses. This is far from comprehensive or refined, and is created to inspire. Other ideas I intend to test include applying this to an agentic system, and training other inputs along with diangosis codes, such as non-medical drivers of health. 

What does this all mean? There is a clear distinction between traditional machine learning models, which are typically designed to predict a limited set of predefined outcomes, and this large language model (LLM). The LLM is capable of generating predictions for any number of outcomes, as long as they are provided—for example, diagnosis codes in this context.


# Data

There are many types of healthcare data, with this example we use synthetic claims data found on CMS's website. The claims data in its original format cannot be used, so we have to conver it into a sequence of diagnosis, or 'sentences'. Note - another idea to test would be to create episodes or encounters based on groups of diagnosis, to mimic paragraphs or sentences in words, and train our llm to recognize macro and micro patterns. Too many ideas so little time.

We first use a simple function to create a ordered sequence of diagnosis per patient and return those sequences as strings. We don't have to focus exactly on tokenization because we dont have punctuation, stemming, or stop words. We treat each diagnosis (dx) as a discrete chunk of words.

```python

dx_sequences = create_sequences(claims_renamed)

```

```python
['1970 6186 29623 3569', '33811 V5789 49121 7366', '42789 5781']
```

We follow standard preprocessing steps such as converting our diagnosis codes into numeric vectors.

``` python

BATCH_SIZE = 32 # default - how many observations per batch that are fed into our neural network
VOCAB_SIZE = 10000 # only consider top 10000 dx by volume
MAX_LEN = 50 # max sequence length

# Create a vectorisation layer
vectorize_layer = layers.TextVectorization(
    standardize="lower", # This converts our text to lowercase, note some dx contain strings. 
    max_tokens=VOCAB_SIZE, # gives the most prevalent dx an integer token
    output_mode="int",
    output_sequence_length=MAX_LEN + 1, # max length of each of our sequences + 1
)

```

This is the end result. We consider up to 50 sequential diagnosis codes, padding any with 0s to indicate the string has come to an end.

```python
# original dx sequence
 1970 6186 29623 3569
# token representation after appling vectorization layer
 [ 222 1112  377  725    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0]

```

To train our decoder-only transformer, we first apply the prepare_inputs() function to structure the dataset for autoregressive learning. This setup enables the model to learn iteratively—generating one token at a time based solely on previously seen tokens.

The function tokenizes each sequence, then shifts it to create input–target pairs:

* The input x contains all tokens up to position i.

* The target y is the token at position i+1.

In effect, the model is trained to predict the next token, having seen everything up to that point. For example:

* It begins with token 22 and learns to predict 1112.

* Then, given the sequence [22, 1112], it predicts 377.

* And so on—each prediction builds recursively on the last.

This approach mirrors how the model will behave at inference time, progressively constructing sequences one token at a time from left to right.

```python
train_ds = sequence_ds.map(prepare_inputs)
```

```python
First Input (x): [ 222 1112  377  725    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0]
First Target (y): [1112  377  725    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0]

```

Source: Medicare Synthetic Claims Data found on *[CMS website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files)*


# Appendix
