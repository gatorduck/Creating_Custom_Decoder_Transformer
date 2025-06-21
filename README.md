# About

My motivation is inspired by current large language models trained on sequences of tokens representing natural language. In this case, each diagnosis code (like an ICD-10 code) is treated as a token in a sequence, just like a word in a sentence. But instead of natural language, the sequence represents a patient’s chronological medical history. The goal of this custom decoder is to learn temporal patterns or predict future diagnoses. This is far from comprehensive or refined, and is created to inspire. Other ideas I intend to test include applying this to an agentic system, and training other inputs along with diangosis codes, such as non-medical drivers of health. 

What does this all mean? There is a clear distinction between traditional machine learning models, which are typically designed to predict a limited set of predefined outcomes, and this large language model (LLM). The LLM is capable of generating predictions for any number of outcomes, as long as they are provided—for example, diagnosis codes in this context.

# Data

There are many types of healthcare data, with this example we use synthetic claims data found on CMS's website. The claims data in its original format cannot be used, so we have to conver it into a sequence of diagnosis, or 'sentences'. Note - another idea to test would be to create episodes or encounters based on groups of diagnosis, to mimic paragraphs or sentences in words, and train our llm to recognize macro and micro patterns. Too many ideas so little time.

We first use a simple function to create a ordered sequence of diagnosis per patient and return those sequences as strings.

```python

dx_sequences = create_sequences(claims_renamed)
```



```python
['1970 6186 29623 3569', '33811 V5789 49121 7366', '42789 5781']
```



Source: Medicare Synthetic Claims Data found on *[CMS website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files)*


# Appendix
