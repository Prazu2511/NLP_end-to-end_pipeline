# NLP_end-to-end_pipeline
This project implements an end-to-end Natural Language Processing (NLP) pipeline for multi-label classification and domain-knowledge entity extraction. It also includes a REST API for model interaction. The key steps in the pipeline are dataset creation, preprocessing, model training, evaluation, extracting entities and connecting through API.

### PROBLEM STATEMENT : 

1. Building and end-to-end NLP pipeline by:
2. Creating multi-column dataset and performing multi-label classification
3. Extracting domain specific entities by combining knowledge base lookup and advanced extraction techniques Providing REST API

### TASK_1:

#### 1.1 Data Creation:
* Rule-based synthetic data generation approach is used in producing a dataset with 1,200 rows.
* Multi-class labels created using where 600 rows are created by rule-based.
* The remaining rows are augmented using “Random Swap Data Augmentation”technique to introduce variability to the data.
* Randomized values assigned to 200 rows to reduce overfitting.
* The final dataset was saved as “multiclass_calls_dataset_with_rand.csv”

#### 1.2 Data Augmentation & Pre-Processing:
* Random Swap Augmentation technique is over textual dataset where it aims to introduce variability into the data, while preserving it’s semantics. This creates slightly altered versions of the original text.
```python
def random_swap(text, n=1):
    words = word_tokenize(text)
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)
```


* Data preprocessing techniques, involving removing of stopwords and lemmatization of the word is done just before training the model and stored in the dataset of "calls_dataset_with_preprocessed.csv"


#### 1.3 Training the Model:
* Two classification approaches were employed for multi-label classification but finally trained on Neural Network model:
    
*Logistic Regression with OneVsRestClassifier:*
  * OneVsRestClassifier wraps around Logistic Regression to handle multi-label classification by training one logistic regression model per label but resulted in overfitting.

```python
 from sklearn.multiclass import OneVsRestClassifier
 model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
```

*Feed Forward Neural Network:*
  
(i) The dataset (multiclass_calls_dataset_with_rand.csv) where the text labels are converted into numerical features using TF-IDF vectorizer
  
```python
from sklearn.feature_extraction.text import TfidfVectorizer
                 
```

(ii) The MultiLabelBinarizer() converts the labels column (strings) into a binary matrix suitable for multilabel classification i.e. Each label is split by ',' to form a list of             labels for each instance.
```python
mlb = MultiLabelBinarizer()
  ```
  
(iii) Splitting the dataset into 80% training set and 20% testing set.
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

(iv) A fully connected neural network is used:
* Input layer, two hidden layers with 512 neurons and ReLU activation function, and an Output layer with sigmoid function.
* Dropout layer is used to prevent overfitting by randomly deactivating 50% of neurons during training.
```python
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
```
  
(v) Compilation & Training of the model:
* Adam optimizer with learning rate 0.0001 is chosen for it’s adaptive learning capability.
* Binary Cross Entropy Loss function is used, which is ideal for multi-label classification tasks.
* Early stopping Monitors the validation loss (val_loss). Stops training if the loss doesn't improve for 2 consecutive epochs (patience=2).
```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop])
```

(vi) Cross-Validation:
image


#### 1.4 Evaluation of model:
* Validation loss (binary_crossentropy).
* Validation accuracy.
```python
loss, accuracy = model.evaluate(X_val, y_val)
```



### TASK_2:

#### Entity/Keyword Extraction 
* The entities of the labels are extracted.
  
* domain_knowledge.json:
```json
{
"competitors": ["CompetitorX", "CompetitorY", "CompetitorZ", "TechCorp", "MediPlus"],
"features": ["analytics", "AI engine", "data pipeline", "automation", "remote monitoring, fast delivery, high resolution"],
"pricing_keywords": ["discount", "budget", "pricing model", "subscription cost"],
"security_keywords": ["SOC2 certified", "data compliance", "encryption", "privacy policy"]
}
  ```

* Using an Advanced extraction technique i.e rule-based approach using regex and recognizing patterns
```python
def extract_with_regex(text):
    patterns = [
        r"\\b(?:CompetitorX|CompetitorY|CompetitorZ)\\b",  # Competitors
        r"\\b(?:analytics|AI engine|data pipeline)\\b",    # Features
    ]
```

* Combining both the approaches resulting in final set of extracted entities.
  
* These are further stored in the dataset called "extracted_entities_dataset".


### TASK_3:



       
  











