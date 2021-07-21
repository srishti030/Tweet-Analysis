# Importing different libraries

import re
import conda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
import wordcloud
from wordcloud import WordCloud
warnings.filterwarnings("ignore", category=DeprecationWarning)
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

#Removing twitter handels(@user)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


#main function
def main():
    #Reading the train and test dataset
    train  = pd.read_csv('/Users/admin/Desktop/PycharmProjects/Tweet-Analysis/DataSet/train.csv')
    test = pd.read_csv('/Users/admin/Desktop/PycharmProjects/Tweet-Analysis/DataSet/test.csv')

    #Check the train dataset
    train.head()

    #Combine training and test data set
    combi = train.append(test, ignore_index=True)

    combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

    # Remove special characters, numbers, punctuations
    combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

    # Removing Short words here we remove 3 and less length words.
    combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    # Overview of data after preprocessing
    combi.head(31974)

    # Tokenization
    tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
    tokenized_tweet.head()

    # Stemming
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])  # stemming
    tokenized_tweet.head()

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    combi['tidy_tweet'] = tokenized_tweet
    combi.info()

    # Differentiate the most and less frequent words in the data set
    all_words = ' '.join([text for text in combi['tidy_tweet']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Words in non racist tweets
    normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # Words in racist tweets
    negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
    wordcloud = WordCloud(width=800, height=500,
                          random_state=21, max_font_size=110).generate(negative_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    # To check the impact of hashtags in Sentiment Analysis
    # function to collect hashtags
    def hashtag_extract(x):
        hashtags = []
        # Loop over the words in the tweet
        for i in x:
            ht = re.findall(r"#(\w+)", i)
            hashtags.append(ht)

        return hashtags

    # extracting hashtags from non racist/sexist tweets

    HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

    # extracting hashtags from racist/sexist tweets
    HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

    # unnesting list
    HT_regular = sum(HT_regular, [])
    HT_negative = sum(HT_negative, [])
    HT_regular

    # Plot the top10 hashtags in both class non racist and racist
    # Non-Racist/Sexist Tweets

    a = nltk.FreqDist(HT_regular)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Count", n=10)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x="Hashtag", y="Count")
    ax.set(ylabel='Count')
    plt.show()

    # Racist/Sexist Tweets

    b = nltk.FreqDist(HT_negative)
    e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
    # selecting top 10 most frequent hashtags
    e = e.nlargest(columns="Count", n=10)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=e, x="Hashtag", y="Count")
    ax.set(ylabel='Count')
    plt.show()

    # Extracting Features
    # Bag of words
    bow_vectorizer = CountVectorizer(max_features=1500, stop_words='english')
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

    # Building Model using Logistic Regression
    train_bow = bow[:31962, :]
    test_bow = bow[31962:, :]

    # splitting data into training and validation set
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], test_size=0.30)

    print(xtrain_bow)
    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain)
    pred_lreg = lreg.predict(xvalid_bow)
    # training the model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yvalid, pred_lreg)
    print("Confusion Matrix")
    print(cm)
    print(" ")
    print("Efficiency")
    logistic_score = lreg.score(xvalid_bow, yvalid)
    print(logistic_score)
    accuracy_score(yvalid, pred_lreg)

    naive = naive_bayes.MultinomialNB()
    naive.fit(xtrain_bow, ytrain)
    pred_naive = naive.predict(xvalid_bow)
    com = confusion_matrix(yvalid, pred_naive)
    print("Confusion Matrix")
    print(com)
    print(" ")
    print("Efficiency")
    naive_score = naive.score(xvalid_bow, yvalid)
    print(naive_score)
    accuracy_score(yvalid, pred_naive)

    svm = SVC(random_state=0, kernel="linear")
    svm.fit(xtrain_bow, ytrain)

    pred_svm = svm.predict(xvalid_bow)
    svm_com = confusion_matrix(yvalid, pred_svm)
    print("Confusion Matrix")
    print(svm_com)
    svm_score = svm.score(xvalid_bow, yvalid)
    print(svm_score)

    accuracy_score(yvalid, pred_svm)
    # print(classification_report(yvalid,pred_svm))

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=0)
    clf.fit(xtrain_bow, ytrain)
    pred_mlp = clf.predict(xvalid_bow)
    mlp_com = confusion_matrix(yvalid, pred_mlp)
    print("Confusion Matrix")
    print(mlp_com)
    mlp_score = clf.score(xvalid_bow, yvalid)
    print(mlp_score)
    accuracy_score(yvalid, pred_mlp)
    # print(classification_report(yvalid,pred_mlp))

    left = [1, 2, 3, 4]
    height = [logistic_score * 100, naive_score * 100, svm_score * 100, mlp_score * 100]
    print(logistic_score * 100, naive_score * 100, svm_score * 100, mlp_score * 100)
    tick_label = ['Logistic Regression', 'Naive Bayes', 'SVM', 'Multilayer perceptron']
    plt.bar(left, height, tick_label=tick_label, width=0.3, color=['red', 'green', 'blue', 'violet'])
    plt.xlabel('Algorithms')
    plt.ylabel('Efficiency')
    plt.ylim((90, 100))
    plt.title('Results Comparison')
    plt.show()


if __name__ == "__main__":
    # calling main function
    main()






