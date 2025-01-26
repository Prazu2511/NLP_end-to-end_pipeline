# NLP_end-to-end_pipeline
This project implements an end-to-end Natural Language Processing (NLP) pipeline for multi-label classification and domain-knowledge entity extraction. It also includes a REST API for model interaction. The key steps in the pipeline are dataset creation, preprocessing, model training, evaluation, extracting entities and connecting through API.

### PROBLEM STATEMENT : 

1. Building and end-to-end NLP pipeline by:
2. Creating multi-column dataset and performing multi-label classification
3. Extracting domain specific entities by combining knowledge base lookup and advanced extraction techniques Providing REST API

### TASK_1:

*1.1 Data Creation:*
* Rule-based synthetic data generation approach is used in producing a dataset with 1,200 rows.
* Multi-class labels created using where 600 rows are created by rule-based.
* The remaining rows are augmented using “Random Swap Data Augmentation”technique to introduce variability to the data.
* Randomized values assigned to 200 rows to reduce overfitting.
* The final dataset was saved as “multiclass_calls_dataset_with_rand.csv”

*1.2 Data Augmentation & Pre-Processing:*
* Random Swap Augmentation technique is over textual dataset where it aims to introduce variability into the data, while preserving it’s semantics. This creates slightly altered versions of the original text.
```python
def random_swap(text, n=1):
    words = word_tokenize(text)
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)
```
* Data preprocessing techniques, involving removing of stopwords and lemmatization of the word is done just before training the model.


