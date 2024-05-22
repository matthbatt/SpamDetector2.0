import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import time
import matplotlib.pyplot as plt


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def main():
    parser = argparse.ArgumentParser(description="Spam detector.")
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Shows the occurrence of words as a wordcloud",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Trains and tests the algorithms and gives results in different metrics",
    )
    parser.add_argument(
        "-c",
        "--classify",
        type=str,
        help="Classifies the given text into spam or not spam using word embeddings",
    )

    args = parser.parse_args()

    # Measure time for loading GloVe embeddings
    start_time = time.time()
    # Source : https://github.com/stanfordnlp/GloVe?tab=readme-ov-file
    embeddings_index = load_glove_embeddings('glove.6B/glove.6B.100d.txt')
    print(f"Loaded GloVe embeddings in {time.time() - start_time:.2f} seconds")

    # Load dataset
    mails = pd.read_csv("formatted_spam.csv", encoding="latin-1")
    # mails.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    mails.rename(columns={"v1": "labels", "v2": "message"}, inplace=True)
    mails["label"] = mails["labels"].map({"ham": 0, "spam": 1})
    mails.drop(["labels"], axis=1, inplace=True)

    # We will now create the two subsets:
    # The training set (75% of the data) and the testing set (25% of the data).
    trainIndex, testIndex = list(), list()
    for i in range(mails.shape[0]):
        if np.random.uniform(0, 1) < 0.75:
            trainIndex += [i]
        else:
            testIndex += [i]
    trainData = mails.loc[trainIndex]
    testData = mails.loc[testIndex]
    
    # Reset the indexes in both subsets.
    trainData.reset_index(inplace=True)
    trainData.drop(["index"], axis=1, inplace=True)
    
    testData.reset_index(inplace=True)
    testData.drop(["index"], axis=1, inplace=True)
    
    # We can visualize the most frequent keywords in the spams and
    # do the same for the non-spams.

    # Show word cloud
    if args.show:
        spam_words = " ".join(list(mails[mails["label"] == 1]["message"]))
        spam_wc = WordCloud(width=512, height=512).generate(spam_words)
        plt.figure(figsize=(10, 8), facecolor="k")
        plt.imshow(spam_wc)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    elif args.test:
        # Dictionary to store performance metrics
        models_performances = {}

        # Logistic Regression with Word Embeddings
        start_time = time.time()
        X_train = transform_messages_to_embeddings(trainData["message"], embeddings_index)
        X_test = transform_messages_to_embeddings(testData["message"], embeddings_index)
        print(f"Transformed messages to embeddings in {time.time() - start_time:.2f} seconds")
        
        y_train = trainData["label"]
        y_test = testData["label"]
        """
        start_time = time.time()
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        print(f"Trained Logistic Regression model in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        print(f"Made predictions in {time.time() - start_time:.2f} seconds")
        print()
        
        print("Results for Logistic Regression with Word Embeddings:")
        models_performances['Logistic Regression'] = metrics(y_test, y_pred)
        """

        # Hyperparameter Tuning for Logistic Regression
        start_time = time.time()
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='f1')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Trained Logistic Regression model with hyperparameter tuning in {time.time() - start_time:.2f} seconds")

        # Predictions and Evaluation
        start_time = time.time()
        y_pred = best_model.predict(X_test)
        print(f"Made predictions in {time.time() - start_time:.2f} seconds")

        print("Results for Tuned Logistic Regression:")
        models_performances['Tuned Logistic Regression'] = metrics(y_test, y_pred)
        
        # TF-IDF Classifier
        sc_tf_idf = TFIDFCLassifier(trainData)
        sc_tf_idf.train()
        preds_tf_idf = sc_tf_idf.predict(testData["message"])
        print()
        print("Results for TF x IDF classifier:")
        models_performances['TF x IDF'] = metrics(testData["label"], preds_tf_idf)
        
        # Bag of Words Classifier
        sc_bow = BowClassifier(trainData)
        sc_bow.train()
        preds_bow = sc_bow.predict(testData["message"])
        print()
        print("Results for Bow classifier:")
        models_performances['Bag of Words'] = metrics(testData["label"], preds_bow)
        print()
        
        # Plotting the metrics
        plot(models_performances)

    elif args.classify:
        # Train on entire dataset
        start_time = time.time()
        X = transform_messages_to_embeddings(mails["message"], embeddings_index)
        y = mails["label"]
        print(f"Transformed messages to embeddings in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        print(f"Trained Logistic Regression model in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        pm_embedding = transform_messages_to_embeddings([args.classify], embeddings_index)
        print(f"Transformed single message to embeddings in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        print("Spam? :", model.predict(pm_embedding)[0])
        print(f"Classified single message in {time.time() - start_time:.2f} seconds")

    else:
        parser.print_help()
        print()
        print("Please select mode.")


def load_glove_embeddings(file_path):
    """
    This function loads a pre-trained GloVe word embeddings.
    Using GloVe embeddings captures the semantic meaning of words.
    It also enables the model to understand contextual similaritires improving the classification accuracy.
    """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def process_message_lemmatizer(message):
    """
    This function is responsible for preprocessing the text data:
        - lemmatization
        - lowercasing
        - tokenization
        - removing non-alphanumeric characters, stopwords
    """
    lemmatizer = WordNetLemmatizer()
    message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if w.isalnum()]
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    return words

def process_message(message, lower_case=True, stem=True, stop_words=True, gram=1):
    """
    Cette fonction est très importante car c'est elle qui transforme les messages
    en une liste de mots clés essentiels: non stop et "stemmés".
    Si gram > 1 ce ne sont pas des mots clés mais des couples de mots clés qui sont
    pris en compte
    """
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [" ".join(words[i : i + gram])]
        return w
    if stop_words:
        sw = stopwords.words("english")
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

def message_to_embedding(message, embeddings_index, embedding_dim=100):
    """
    Converts a text message into an embedding vector.
    """
    words = process_message(message)
    embedding_matrix = []
    for word in words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix.append(embedding_vector)
    if len(embedding_matrix) == 0:
        return np.zeros(embedding_dim)
    else:
        return np.mean(embedding_matrix, axis=0)


def transform_messages_to_embeddings(messages, embeddings_index, embedding_dim=100):
    embeddings = [message_to_embedding(message, embeddings_index, embedding_dim) for message in messages]
    return np.array(embeddings)


class SpamClassifier:
    def __init__(self, trainData):
        self.mails, self.labels = trainData["message"], trainData["label"]

    def train(self):
        pass

    def classify(self, message):
        pass

    def calc_TF_and_IDF(self):
        noOfMessages = self.mails.shape[0]
        self.spam_mails, self.ham_mails = (
            self.labels.value_counts()[1],
            self.labels.value_counts()[0],
        )
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()

        for i in range(noOfMessages):
            message_processed = process_message(self.mails.get(i))
            count = list()
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count.append(word)
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def predict(self, testData):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result


class TFIDFCLassifier(SpamClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()

    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = self.tf_spam[word] * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam[word] + self.idf_ham.get(word, 0))
            )
            self.sum_tf_idf_spam += self.prob_spam[word]

        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                self.sum_tf_idf_spam + len(self.prob_spam.keys())
            )

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log(
                (self.spam_mails + self.ham_mails)
                / (self.idf_spam.get(word, 0) + self.idf_ham[word])
            )
            self.sum_tf_idf_ham += self.prob_ham[word]

        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (
                self.sum_tf_idf_ham + len(self.prob_ham.keys())
            )

        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(self.prob_spam.keys()))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.sum_tf_idf_ham + len(self.prob_ham.keys()))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam


class BowClassifier(SpamClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.calc_TF_and_IDF()
        self.calc_prob()

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (
                self.spam_words + len(self.tf_spam.keys())
            )
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (
                self.ham_words + len(self.tf_ham.keys())
            )
        self.prob_spam_mail, self.prob_ham_mail = (
            self.spam_mails / self.total_mails,
            self.ham_mails / self.total_mails,
        )

    def classify(self, processed_message):
        pSpam, pHam = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.spam_words + len(self.prob_spam.keys()))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.ham_words + len(self.prob_ham.keys()))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return pSpam >= pHam


def metrics(labels, predictions):
    # Ensure labels are a numpy array
    labels = np.array(labels)
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (
        true_pos + true_neg + false_pos + false_neg
    )

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy:", accuracy)

    return [precision, recall, Fscore, accuracy]


def plot(model_info, filename='performance_comparison.png'):
    models = list(model_info.keys())
    metrics = list(model_info.values())

    precision = [m[0] for m in metrics]
    recall = [m[1] for m in metrics]
    fscore = [m[2] for m in metrics]
    accuracy = [m[3] for m in metrics]

    x = range(len(models))

    plt.figure(figsize=(12, 6))

    bar_width = 0.2
    r1 = np.arange(len(precision))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.bar(r1, precision, color='b', width=bar_width, edgecolor='grey', label='Precision')
    plt.bar(r2, recall, color='g', width=bar_width, edgecolor='grey', label='Recall')
    plt.bar(r3, fscore, color='r', width=bar_width, edgecolor='grey', label='F-score')
    plt.bar(r4, accuracy, color='c', width=bar_width, edgecolor='grey', label='Accuracy')

    plt.xlabel('Models', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(precision))], models)
    plt.title('Performance Comparison of Different Classifiers')
    plt.ylim(0, 1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    main()