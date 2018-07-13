# JDChat   
### Information Retrieval (Sub model for Reinforcement Learning)
### Environment
> - python 2.7  
> - tensorflow 1.8.0
### Files
  
]
> - Data Preporcessing --------- Data.py Data_Visualization.ipynb
> - TFIDF model ---------------------- TFIDF.py
> - LM model   ---------------------- LM.py
> - CNN model  ---------------------- CNN_model.py CNN_train.py
> - Inference ------------------ Inference.py

### Data preprocessing
> - Address origin data (Chinese stop words removal, QA-pair --> pred_QA-pair.csv)
> - Generate CNN data (padding sentences 32 words each sentence and word embedding)

### Models

#### TFIDF
> 1. Data preprocessing and read pred data
> 2. Generate tfidf representation for each sentence
> 3. Caculate cosine similarity for question - question pair
> 4. Get the answer with the most similar question

#### LM
> 1. Data preprocessing and read pred data
> 2. Caculate LM similarity for question - question pair
> 3. Get the answer with the most similar question

#### CNN
> 1. Data preprocessing and read pred data
> 2. Generate word embedding (initialize with baidu baike vector) representation
> 3. Train the CNN model with question - answer pair
> 4. Feed the question - answer pair and Get the most similar answer

#### Inference
> - Input : question, top_k
> - Output : TFIDF LM CNN top k response

*plus: Data not uploaded beacuse of the privacy*

