# About

My motivation is inspired by current large language models trained on sequences of tokens representing natural language. In this case, each diagnosis code (like an ICD-10 code) is treated as a token in a sequence, just like a word in a sentence. But instead of natural language, the sequence represents a patient’s chronological medical history. The goal of this custom decoder is to learn temporal patterns or predict future diagnoses. This is far from comprehensive or refined, and is created to inspire. Other ideas I intend to test include applying this to an agentic system, and training other inputs along with diangosis codes, such as non-medical drivers of health.

What does this all mean? This is a stark difference between traditional machine learning models that can predict few or several outcomes. This llm model is limited to predicting as many outcomes as you can provide, or in this case diagnosis codes.

Data: Medicare Synthetic Claims Data found on *[CMS website](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files)*
