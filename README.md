# Email/SMS-Spam

A machine learning project that classifies SMS text messages as **spam** or **ham (not spam)** using NLP preprocessing, TF-IDF vectorization, and a comparison of multiple classification algorithms — culminating in a Multinomial Naive Bayes model chosen for its high precision.

## Business Problem

Spam messages are a nuisance at best and a phishing/fraud risk at worst. A reliable spam filter needs to satisfy one non-negotiable constraint: **it must almost never mark a real (ham) message as spam**, even if that means letting a few spam messages slip through. This makes **precision** — not just accuracy — the metric that matters most for this problem, and it directly shaped the model selection below.

## Dataset

- **Source**: `spam.csv` (SMS Spam Collection dataset, latin-1 encoded)
- **Raw size**: 5,572 messages, 5 columns (3 mostly-empty unnamed columns dropped during cleaning)
- **After cleaning**: 5,169 messages (403 duplicates removed), 2 columns (`target`, `text`)
- **Class balance**: 4,516 ham (87.4%) vs. 653 spam (12.6%) — a meaningfully imbalanced dataset


## Tech Stack

- **Python** — pandas, numpy
- **NLP** — nltk (tokenization, stopword removal, Porter stemming), wordcloud
- **Feature Engineering** — scikit-learn (`TfidfVectorizer`, `MinMaxScaler`)
- **Modeling** — scikit-learn (Naive Bayes, SVM, Logistic Regression, tree/ensemble methods, voting & stacking classifiers)
- **Visualization** — matplotlib, seaborn

## Methodology

1. **Data Cleaning** — dropped unused columns, renamed `v1`/`v2` to `target`/`text`, label-encoded the target (ham=0, spam=1), removed 403 duplicate rows.
2. **Exploratory Data Analysis** — class distribution, and message-length statistics (characters, words, sentences) split by class. Spam messages are notably longer: ~138 characters and ~28 words on average vs. ~70 characters and ~17 words for ham.
3. **Text Preprocessing** — a custom `transform_text()` pipeline: lowercase → tokenize → keep alphanumeric tokens only → remove stopwords and punctuation → Porter stemming. Word clouds and frequency plots were generated separately for spam and ham vocabularies to sanity-check the transformation.
4. **Feature Extraction** — `TfidfVectorizer` (top 3,000 features) applied to the cleaned text, with `MinMaxScaler` normalization.
5. **Model Comparison** — 11 classifiers were trained and evaluated on an 80/20 train-test split, scored on both accuracy and precision (given the class imbalance and the cost of false positives):

   |   Algorithm                 | Accuracy    | Precision |
   
   | Random Forest               |   0.976     | **0.983** |
   | K-Nearest Neighbors         |   0.905     |   0.976   |
   | Extra Trees                 |   0.975     |   0.975   |
   | Logistic Regression         |   0.967     |   0.964   |
   | XGBoost                     |   0.970     |   0.950   |
   | **Multinomial Naive Bayes** |   0.979     |   0.946   |
   | SVM (sigmoid kernel)        |   0.969     |   0.927   |
   | Gradient Boosting           |   0.947     |   0.919   |
   | Bagging Classifier          |   0.958     |   0.868   |
   | AdaBoost                    |   0.925     |   0.849   |
   | Decision Tree               |   0.932     |    —      |

6. **Ensembling** — a soft-voting classifier (SVM + Naive Bayes + Extra Trees) and a stacking classifier (same base learners, Random Forest meta-learner) were also tested, reaching 0.980 and 0.979 accuracy respectively, without clearly beating simpler models on precision.
7. **Model Selection** — **Multinomial Naive Bayes** was selected as the final model. Despite Random Forest posting the single highest precision score, MultinomialNB offers a strong, consistent balance of accuracy (97.9%) and precision (94.6%), pairs naturally with TF-IDF text features, and is lightweight and fast enough for real-time inference — a good fit for a text classification task like this.
8. **Serialization** — the fitted `TfidfVectorizer` and final `MultinomialNB` model were pickled (`vectorizer.pkl`, `model.pkl`) for reuse in a downstream application.

## Key Results

- **Final model**: Multinomial Naive Bayes on TF-IDF features (max 3,000 features)
- **Accuracy**: 97.9%
- **Precision**: 94.6% (of messages predicted as spam, ~95% actually are — minimizing false positives on real messages)
- **Confusion Matrix** (test set, n=1,034):

  |                 | Predicted Ham   | Predicted Spam |
  | **Actual Ham**  |      889        |       7        |
  | **Actual Spam** |       15        |      123       |



## License

This project is available under the MIT License.
