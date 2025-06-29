# About

My motivation is inspired by current large language models trained on sequences of tokens representing natural language. In this case, each diagnosis code (like an ICD-10 code) is treated as a token in a sequence, just like a word in a sentence. But instead of natural language, the sequence represents a patient’s chronological medical history. The goal of this custom decoder is to learn temporal patterns or predict future diagnoses. This is far from comprehensive or refined, and is created to inspire. Other ideas I intend to test include applying this to an agentic system, and training other inputs along with diangosis codes, such as non-medical drivers of health. 

What does this all mean? There is a clear distinction between traditional machine learning models, which are typically designed to predict a limited set of predefined outcomes, and this large language model (LLM). The LLM is capable of generating predictions for any number of outcomes, as long as they are provided—for example, diagnosis codes in this context.


# Data

There are many types of healthcare data, with this example we use synthetic claims data found on CMS's website. The claims data in its original format cannot be used, so we have to conver it into a sequence of diagnosis, or 'sentences'. Note - another idea to test would be to create episodes or encounters based on groups of diagnosis, to mimic paragraphs or sentences in words, and train our llm to recognize macro and micro patterns. Too many ideas so little time.

### Tokenization

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
### Embeddings

Once again we convert our numbers into more numbers, but these are high dimensional numbers that encode meaning of a word or in our case diagnosis. This includes positional encoding.

```python

MAX_LEN = 50 # max sequence length
VOCAB_SIZE = 10000 # only consider top 10000 dx by volume
EMBED_DIM = 2 # embedding size for each token

embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM)

```

```python
Token 222: [ 0.05086685 -0.05951506]
Token 1112: [-0.00821701 -0.00422239]
Token 377: [ 0.06296659 -0.076947  ]
Token 725: [-0.06489329 -0.01285102]

```

# Training

Next step uses Keras functional approach to build our model, the first several layers are created to build the framework. 

```python
# encapsulated in a function def(model)

# Functional API approach 
    inputs = layers.Input(shape=(MAX_LEN,), dtype="int32") # input layer
    embedding_layer = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBED_DIM) # embedding layer
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FEED_FOWARD_DIM) # attention layer + hidden layer (neural network)
    x = transformer_block(x)
    outputs = layers.Dense(VOCAB_SIZE)(x) # output layer - dependent on mutliclass classification, in this case all possible outcomes or our total vocab (dx) size
    model = models.Model(inputs=inputs, outputs=[outputs, x]) # create our model, specify which inputs and outputs to use, within our outputs we are returning two sets of data, outputs which is, and x which are our attention scores

```

### Transformer

The secret sauce to our model is in the TransformerBlock which helps us create masked self-attention values. This piece uses a subclassing API approach where we define layers and then use them by using a call method all within our TransformerBlock object.

The first lines before our call() method define our steps. The call method executes them.

```python

class TransformerBlock(layers.Layer):
    def __init__(self, EMBED_DIM, NUM_HEADS, FEED_FOWARD_DIM, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(NUM_HEADS, EMBED_DIM)
        self.ffn = keras.Sequential( 
            [
                layers.Dense(FEED_FOWARD_DIM, activation="relu"),
                layers.Dense(EMBED_DIM),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
(1)     attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

```


All the action occurs in our MultiHeadAttentionLayer (1). This is where we calculate query, key, and value. To keep it simple,, lets observe our sequence  [1970 6186 29623 3569] or token [ 222 1112  377  725 ...0 0 0 0].

1. Create a QUERY per token by taking the dot product of our original position encoded embeddings and weights. I like to think of these as a reference vector.
    
     $$\vec{e}_{i}  W = \vec{q}_i $$

    $$ \ \begin{bmatrix} 0.05086685 \\ -0.05951506 \end{bmatrix} *
    \ \begin{bmatrix}
       \frac{}{} & \frac{}{} &            \\[0.3em]
       \frac{}{} &    W eights       & \frac{}{} \\[0.3em]
                  & \frac{}{} & \frac{}{}
     \end{bmatrix} = \begin{bmatrix} {} \\ {Query} \\ {} \end{bmatrix} $$

    This should return 2 new numbers per token as we originally set our embedding size to 2.

2. We do the same and multiply our embeddings against KEY specific weights and return KEY values.

     $$\vec{e}_{i}  W = \vec{k}_i $$

    $$ \ \begin{bmatrix} 0.05086685 \\ -0.05951506 \end{bmatrix} *
    \ \begin{bmatrix}
       \frac{}{} & \frac{}{} &            \\[0.3em]
       \frac{}{} &    W eights       & \frac{}{} \\[0.3em]
                  & \frac{}{} & \frac{}{}
     \end{bmatrix} = \begin{bmatrix} {} \\ {Key} \\ {} \end{bmatrix} $$


3. We then take the dot product of our QUERY and KEY transposed. This calculates our similarities between our token of interest (as a QUERY) and our corresponding tokens in our sequence of diagnosis (KEYS). Note because this a decoder transformer we masks our values. The higher the similarity scores are the more our keys and querys align. Softmax is calculated for each of these in order to return attention probabilities.


$$ E K^{T} = similarity \ scores$$
$$ \begin{bmatrix} {} \\ {Query} \\ {}  \end{bmatrix}   \begin{bmatrix} && {Key}  && \end{bmatrix} = \begin{bmatrix} {x} & {x} & {x} \\ {} & {x} & {x} \\ {} & {} & {x} \end{bmatrix} + Softmax
$$

4. VALUE also calculated using dot product and VALUE specific weights.

     $$\vec{e}_{i}  W = \vec{v}_i $$

    $$ \ \begin{bmatrix} 0.05086685 \\ -0.05951506 \end{bmatrix} *
    \ \begin{bmatrix}
       \frac{}{} & \frac{}{} &            \\[0.3em]
       \frac{}{} &    W eights       & \frac{}{} \\[0.3em]
                  & \frac{}{} & \frac{}{}
     \end{bmatrix} = \begin{bmatrix} {} \\ {Value} \\ {} \end{bmatrix} $$

5. Last piece multiplies our probabilities against our values for each diagnosis.This then added to enrich our original embedding with influence from neighboring tokens.

Another simpler way to wrap your head around this can be through looking at a multiplication table of our query, key, and value. 

Upon completion of training, we can supply a new input token to generate the next predicted diagnosis code—potentially even producing both the immediate next token and the subsequent one.


```python
y_pred, attention_scores = model.predict(x, verbose=0)
```

```python
Diagnosis Code: 486, Probability: 4.0302
Diagnosis Code: v5789, Probability: 3.8886
Diagnosis Code: 4280, Probability: 3.8728
Diagnosis Code: 0389, Probability: 3.7248
Diagnosis Code: 49121, Probability: 3.6444
```

This is just the beginning incorporate more information outside sequences of diagnosis such as chronic conditions or non-medical drivers of health that are also run in our neural network however as separate input that with its own individual layers and subsequentaly further enrich our diagnosis embeddings. 

# Appendix

Data Source: Medicare Synthetic Claims Data found on *[CMS website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files)*
