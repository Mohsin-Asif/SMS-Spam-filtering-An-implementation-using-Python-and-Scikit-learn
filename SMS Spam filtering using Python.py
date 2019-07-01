
# coding: utf-8

# # <font color = "581845"><b>Creating a SMS Spam Filter: Using ML and Data Mining Algorithms to Detect, Train, and Filter spam</b> </font>
# #### <i><font color = "grey">Krystal Tyhulski, Mohsin Asif, and Colton Zwissler</font></i>

# #### <center> <font color = 581845>Abstract</font></center>
# Spam is becoming a growing concern for SMS users around the world. SMS spam can easily target and impact users without deception if the user has a limited plan and the message incurs a fee. This study creates an SMS spam filter by using machine learning algorithms to detect spam. We will use a natural language toolkit (NLTK) for text processing, Term Frequency-Inverse Document Frequency (TF-IDF) for the conversion of text data, and Logistic Regression, Naïve Bayes, Support Vector Machines (SVM) for the text classification machine learning algorithms. In addition to determining accuracy, we will use a classification matrix to examine the results based on the parameter of precision and use an AUC (Area Under the Curve) ROC (Receiver Operating Characteristics) curve to measure the overall performance of each model.

# ## <font color = 581845>Introduction </font>

# ### <font color = 900C3F>Background </font>
# 
# Spam, unsolicited and, often times, inappropriate messages sent electronically, was once largely received via email. However, as technology has improved in filtering out these emails and user awareness has increased, malicious messages are growing across a new channel: Short Messaging Service (SMS) aka text message [3].
# 
# Spammers are pivoting to this new medium due to the increased availability of inexpensive, prepaid SMS plans. In Russia, bots are being used to simulate real users and to send seemingly authentic messages [9], and at this point, SMS is still considered a trustworthy method of communication by users, making them more susceptible to malicious texts [1]. This problem has been growing in Asia since the early 2000s in countries like China, India, and Pakistan with Chinese mobile phone users receiving an average of 10.35 spam messages per week in 2008. On the other hand, this issue is still gaining traction in the US with spam accounting for less than 1% of SMS messages in 2010, compared to 30% in parts of Asia [8].
# 
# ### <font color = 900C3F>Problem Statement </font>
# These days, it is difficult to look around and find someone without a phone in their hand. Mobile devices are now an essential part of everyday life and the primary means of communication for people in most countries, with billions of text messages being sent around the world every day. Since text messages are typically very short and often written with numerous abbreviations, they are more challenging for filters to detect. Without an effective method for filtering out spam sent through SMS, mobile users will be at an increasingly high risk of being victimized by these malicious attacks [5]. This study will aim to create an SMS spam filter that uses machine learning algorithms to train, predict, and effectively detect spam.
# 
# ### <font color = 900C3F>Research Questions </font>
# The following research questions will be addressed throughout this study:
# 1.	Is the length of a message an indicator as to whether or not a message is “ham” or “spam?”
# 2.	Are there specific words used more frequently in “ham” or “spam” messages?
# 3.	What machine learning algorithm provides the most accurate results for SMS spam detection?
# 
# ### <font color = 900C3F>Literature Review </font>
# Identifying spam with machine learning for filtering can generally be split into three components. These components include text processing, the formation of a feature vector, and text classification machine learning algorithms [7].
# 
# #### <font color = 'C70039'>Text Processing </font>
# Cleaning up large amounts of unnecessary text plays a crucial part in classifying spam. Reducing the amount of text through removing redundant words or stop words through normalization, or converting words to their root through stemming, or using tokenization to simply separate letters and punctuation help dissect SMS messages. The number of words in a text classification can be quite large, so it’s important to understand how to convert text data into high dimensional vectors to help decrease the dimension of data [7].
# 
# #### <font color = 'C70039'>Machine Learning Features for SMS Spam Filtering </font>
# In order to avoid problems with text classification, it is necessary to use feature extraction and feature selection. With feature extraction, the raw dataset is transformed into a reduced dimensional space, whereas with feature selection, the unwanted features are removed through filters, wrappers and embedded methodologies [12]. Some common techniques for feature extraction/selection for spam include the following:
# 
# #### <font color = '5D6D7E'><i>Bag of Words</i> </font>
# A simple method of representing text data by creating a list of all unique words that occur in a collection of documents through extracting features for later use in machine learning algorithms [2].
# 
# #### <font color = '5D6D7E'><i>CountVectorizer</i> </font> 
# Converts a collection of text documents into a matrix of integers which helps create a sparse matrix of the counts [11].
# 
# #### <font color = '5D6D7E'><i>Information Gain</i> </font> 
# A key that is used by a Decision Tree algorithm which measures the reduction in entropy of the class after a feature is selected, or how common the feature is observed in that class compared to others [4].
# 
# #### <font color = '5D6D7E'><i>Latent Dirichlet Allocation</i> </font> 
# A generative statistical model for collections of distinct data that allows observations to be explained by unobserved groups that can describe which parts of the data are similar [4].
# 
# #### <font color = '5D6D7E'><i>TF-IDF Vectorization</i> </font> 
# Term Frequency-Inverse Document Frequency (TF-IDF), is a popular term-weighting method that looks at the importance of a word in a collection of documents. The approach creates a matrix from the messages, then calculates the frequency of a word, and the greater the occurrence of a word in a document the higher the value [11].
# 
# #### <font color = 'C70039'>Machine Learning Algorithms for SMS Spam Filtering </font>
# Many studies on spam filtering have used both feature extraction and feature selection to help reduce data dimensionality. In return, the objective is to improve the performance and computational efficiency of machine learning algorithms that are in use for predicting the SMS spam messages [2]. Some common types of supervised machine learning algorithms for text classification with spam include the following:
# 
# #### <font color = '5D6D7E'><i>Decision Trees</i> </font> 
# A data mining practice that is constructed from a top node – the “root” to bottom nodes – the “leaves,” and represent decisions for decision making to predict or classify unknown data. Two common types of methods generated from decision trees are ID3 (Iterative Dichotomiser 3) and C4.5. ID3 supports categorical attributes, whereas C4.5 supports both categorical and numerical attributes [10].
# 
# #### <font color = '5D6D7E'><i>K-Nearest Neighbors (k-NN)</i> </font> 
# As part of instance-based learning, k-NN chooses random data points as seeds and then assigns the seeds to a cluster or feature space whose center is located at the closest distance to predict the classification of the data points [8].
# 
# #### <font color = '5D6D7E'><i>Logistic Regression</i> </font> 
# A statistical model used for binary classification, logistic regression is based upon a dichotomous variable, or problems with two possible outcomes with values usually set as 1 or 0 [9].
# 
# #### <font color = '5D6D7E'><i>Naïve Bayes</i> </font> 
# Originated from Bayesian learning, Naïve Bayes sets probabilistic classifiers through learning parameters by detecting features independently, regardless of other features from the dataset and derives statistics for classifications [9].
# 
# #### <font color = '5D6D7E'><i>Random Forests</i> </font> 
# An ensemble of decision trees that are used to produce predictions that are different from other decision trees. The results can be generalized through averaging them, and they help to handle the issue of overfitting which can affect the reliability of data predictions [9].
# 
# #### <font color = '5D6D7E'><i>Support Vector Machines (SVM)</i> 
# </font> 
# Used for classification and regression analysis, SVMs recognize patterns through data analysis. A set of training examples are mapped as data points in space so that examples of separate categories are divided by a wide gap to find the maximum distance to classify the data points [8].

# ## <font color = 581845>Methodology</font>
# ### <font color = 900C3F>Architecture </font>
# The research methodology used in this study is introduced in the following outline:
# 1.	Acquire the dataset
# 2.	Prepare the corpus 
# 3.	Process the text with NLTK and then analyze
# 4.	Split the data into training and testing datasets
# 5.	Balance the dataset
# 6.	Perform vectorization using TF-IDF Vectorizer
# 7.	Feed to the machine learning classifiers Naïve Bayes, Support Vector Machines, and Logistic Regression
# 8.	Discuss and compare the overall results of precision and accuracy
# 9.	Determine the best algorithm
# [6]
# 
# ### <font color = 900C3F>Justification for Methodology</font>
# In conjunction with our methodology, we first need to understand and explain why we decided to choose the following techniques for creating our SMS spam filter. First, for text processing, NLTK is the most popular Python library that will provide us with all the tools necessary for what we would like to achieve. Second, for converting text data into numerical matrices, TF-IDF Vectorization will be used as we will not have to create a Bag of Words model or other steps using CountVectorizer since it replaces them. Third, we will use supervised machine learning as opposed to semi-supervised or unsupervised machine learning since we are using a large amount of labeled data for our training dataset to classify SMS spam [1]. We decided that the supervised machine learning algorithms that were most commonly used for SMS spam filtering and were within the scope of our project included Logistic Regression, Naïve Bayes, and Support Vector Machines.

# ## <font color = 581845>Data Acquisition </font>
# 
# The SMS Spam Collection Data Set, one of the largest and publically used SMS spam datasets, is used in this project to train and detect spam messages. The dataset, which was created by Tiago A. Almeida and José María Gómez Hidalgo, was retrieved from the UCI Machine Learning Repository, and includes a corpus containing 5,572 English, raw text SMS messages that has been collected for mobile device and spam research studies. 
# 
# The dataset is compiled from various sources: it incorporates 1,002 ham and 322 spam messages from a publicly available dataset, SMS Spam Corpus v.0.1 Big. Another 3,375 ham messaes were randomly selected from the NUS SMS Corpus dataset, which has over 10,000 ham and spam SMS messages. Rougly 425 SMS spam messages were collected from a United Kingdom website forum called Grumbletext, where users report public claims about spam messages they receive on their devices. Finally, the remaining 450 messages were retrieved from a researcher's PhD thesis. 
# 
# The goal of the SMS Spam Collection Data Set is to serve as a good baseline for researchers to get accuately detect and train SMS spam messages in a way that is comparable to that of real-world spam detection mechanisms. 
# 
# 
# 
# To learn more about the dataset, visit the UCI Machine Learning Repository website in the cell below.    

# In[1]:


from IPython.display import IFrame
IFrame('http://archive.ics.uci.edu/ml/datasets/sms+spam+collection', width=1200, height=300)


# #### <font color = 'C70039'> Loading Key Libraries </font>

# In[149]:


# Import the pandas library to create a dataframe.
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import string

from sklearn.preprocessing import LabelEncoder
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.metrics import roc_curve
from sklearn.svm import SVC


import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#nltk.download('punkt)
#nltk.download('wordnet')

from wordcloud import WordCloud
                
from collections import Counter
import numpy as np

from IPython.display import Image


# # Defining Functions

# In[74]:


def plot_roc_curve(fpr, tpr): 
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show() 


# #### <font color = 'C70039'>Retrieving the Data </font>

# To view the corpus, we start by reading in the SMSSpamCollection file using the line.rstrip() method to remove all while spaces from the right side of each string on each line. We then print the length of total messages in sms_messages, which returns a value of 5,574 documents. 

# In[75]:


# Read in the SMS dataset and assign it to sms_messages.
sms_messages = [line.rstrip() for line in open('SMSSpamCollection')]


# In[76]:


# Print the total number of messages in the dataset. 
print(len(sms_messages))


# The dataset is comprised of two columns separated by tabs (\t). The first column classifies the messages as ham (legitimage, non-spam) or spam, and the second column is the raw text message, meaning no text processing, stop word removal, or any other alterations has been applied to the messages. Because our file already has the spam and ham labels, we can use this to train our our machine learning model to automate spam detection in SMS messages. 
# 
# We need to remove the tabs to be able to read the messages in a more human-friendly way, as demonstrated below.

# In[77]:


# Print the first 3 lines of the file. 
print(sms_messages[0:3])


# In[78]:


# Get a better view of what the messages look like. 
print(sms_messages[0],'\n')
print(sms_messages[1],'\n')
print(sms_messages[2],'\n')
print(sms_messages[3],'\n')


# Pandas is used to separate the tabs and create a dataframe with two columns: 1) "Class", which classifies the messages as "ham" or "spam," and 2) "SMS Message", which shows the actual message. 

# In[79]:


# Create a pandas dataframe.
df =pd.read_csv('SMSSpamCollection', 
                # Separate the messages by the tabs.
                sep='\t',
                # Name the first column "Class" and the second column "SMS Message"
                names = ['Class','SMS Message'])

# Look at the first 10 entries of the dataframe. 
df[0:10]


# In[80]:


# Get the shape of the data frame. 
df.shape


# ## <font color = 581845>Data Analysis </font>
# 
# #### <font color = 'C70039'>Qualitative Analysis </font>
# By looking at the first 10 entries of the dataframe above, we can already begin to identify some distinguishing characteristics about spam messages. At first glance, we can see that spam messages are likely to use words such as "free" (lines 2 and 7) and  "win" (lines 2 and 8) Furthermore, it also seems that the author of spam messages use a time frame or limit in order to get the user to do something (i.e., respond, click a link, use a code, etc.). We can see this in the following messages:
# <ul>
#     <li>3: Free entry in a <b>2 a wkly</b> comp...</li>
#     <li>5. Hey there darling it's been <b>3 weeks</b>...</li>
#     <li>9. Had your mobile <b>11 months</b> or more? </li>
# </ul>
# Let's see if the rest of message 8 includes verbage about some kind of time frame:

# In[81]:


df["SMS Message"][8]


# Finally, message 8's "Valid <b>12 hours</b> only" makes 4/4 spam messages from this small glimps of subdata that use a time frame. None of the ham messages in the 10 above demonstrate this pattern.
# 
# Identifying certain features or characteristics that exist in spam messages is an important way to be able to appropriately classify messages as spam. 
# 
# Next, we take a look at the entire dataframe to get some numerical statistics about each class, then look at each individual class to identify additional features to consider in our machine learning models. 

# #### <font color = 'C70039'>Quantitative Analysis </font>

# In the dataframe, there are 5,169 (out of 5,572) SMS messages (92.76%) are unique. This is due to some messages being duplicates.

# In[82]:


# Get more information about the dataset. 
df.describe()


# There are a total of 4,825 (86.6%) ham messages and 747 (13.4%) spam messages. In terms of frequency, we can see that "Sorry, I'll call you later" was the top ham message and was used 30 times. The top spam message, which starts with "Please call our customer service representative," appears 4 times in our dataset. 
# 
# Though some duplicates exist, they "typically correspond to templates often presented in cell phones, and used in legitimate messages (e.g. "Sorry, I'll call you later")" (Almeida, Gómez, & Yamakami, 2011, p. 2). The spam duplicates are likely sent by the same organization.
# 
# To address the problem of duplicates,the creators of this dataset used plagiarism detection techniques based on String-of-Text to detect "near-duplicates." They conclude that, though duplicates exist, the collection of data "does not lead to near-duplicates that may ease the task of detecting SMS spam" (p. 3). 

# In[83]:


df.groupby('Class').describe()


# To get an visual of how many spam vs. ham messages there are in our dataset, we use matplotlib to create a pie chart. 

# In[84]:


# Create a pie chart to visualize the number of ham vs. spam messages.
df["Class"].value_counts().plot(kind = 'pie', 
                                explode = [0, 0.1], 
                                figsize = (6, 6), 
                                autopct = '%1.1f%%', 
                                shadow = True,
                                colors = ['black', 'darkgray'],
                                textprops={'color':"w"})
plt.title('Ham vs. Spam Messages')
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


# Now that we have some information about the dataset, we need to identify some  features to consider in determining which messages are ham and which ones are spam. The first feature we consider is the length of messages, which leads us to our first research question. 

# ##### <font color="FF5733"> Research Question 1: Is the length of a message an indicator as to whether or not a message is ham or spam? </font>

# To answer this question, we need a way to identify the length of each message, then get a visualization by plotting them in a graph. First, we create a new column, "Length," to count all of the characters in the "SMS Message" column to see if there is a difference in length between spam and ham messages. 

# In[85]:


# Create a new column on the dataframe to show the length of each message.
df['Length'] = df['SMS Message'].apply(len)

# See the length of the first 10 messages. 
df[0:10]


# The average message is 80 characters. The smallest message is 2 characters and the longest message is 910 characters.

# In[86]:


df['Length'].describe()


# In[87]:


# Taking a look at the message with the smallest value (2 characters)
df[df['Length'] == 2]['SMS Message'].iloc[0]


# In[88]:


# Taking a look at the message with the longest value (910 characters)
df[df['Length'] == 910]['SMS Message'].iloc[0]


# #### <font color = 'C70039'>Visualizing the Data </font>

# We create a histrogram by plotting the length to see the frequency of text length. Most of the messages are less than 200 characters. 

# In[89]:


df.Length.plot(bins=20, 
               kind="hist", 
               title="Length of All SMS Messages", 
               color="black")


# Next, we separate the ham from spam messages into two dataframes so that we can get more information about the messages by class (ham and spam).
# 
# The mean of all ham messages is 71.48 characters in length. The smallest non-spam message is 2 characters and the largest non-spam message is 910 characters. The smallest and largest documents in our corpus are ham messages. 
# 
# The mean of all spam messages is 138.67, with the smallest message being 13 characters and the largest spam message containing 223 characters. 

# In[90]:


# Create a datafram containing all ham messages
ham = df[df.Class == "ham"]

# Print the first 5 ham messages. 
print(ham.head())
print("")

# Show statistical data about ham messages. 
print(ham.Length.describe())


# In[91]:


# Create a dataframe of all the spam messages. 
spam = df[df.Class == "spam"]

# Print the first 5 spam messages. 
print(spam.head())
print("")

# Show statistical data about spam messages.
print(spam.Length.describe())


# The histogram below shows the length of ham and spam messages. From the graphs below, we can see that the majority of ham messages are in the range of 70 or so characters, while most spam messages are longer than 200 characters long. This suggests that the length of messages is relevant in detecting and predicting spam messages. 

# In[92]:


df.hist(column='Length', by='Class', bins=60, color="black",figsize = (11,5))


# ### <font color = 900C3F>Text Processing </font>
# 
# Now that we've taken a look at the data, the next step is to clean the data. This  needs to be done because SMS messages are typically shorter and full of slang, symbols, emogis and emoticons, abbreviations, acrymomys, and misspelled words which produce a lot of noise in our dataset. 
# 
# Machine learning algorithms, coupled with normalization, stemming, and tokenization will be used to address some of these issues. 

# #### <font color = 'C70039'>Normalization </font>
# 
# Normalization is defined as "a process that makes something more normal or regular" (Normalization Wiki). When discussing normalization in terms of SMS spam classification and detection, normalization is a process used to transform  or remove text in order to reduce noise and group terms that are similar in semantics. 

# First, we create a new column which will be an encode of Class column. Using LabelEncoder() to normalize the class labels by setting 'ham' messages to equal 0 and 'spam' messages to equal 1. These labels will be used later on in the demonstration. 

# In[93]:


# Use LabelEncoder() to normalize class labels (ham and spam)
le = LabelEncoder()
# Encode the Class column. 
y_enc = le.fit_transform(df['Class'])
# Create a new column called "Label" to show the normalized label.
df["Label"] = y_enc
# Show the first 5 messages of the dataframe to see the results. 
df.head()


# We then remove any extra spaces, line breaks, and tabs. We also need to make all text lowercase because, for example, "hello" and "HELLO" may be seen as two different words.
# 
# I first make a copy of the SMS Message column and place it in a new column called Clean Message. The Clean Message column will store the SMS Message after it has been cleaned to get a visual of how the text processing works. The SMS Message column will remain the original, raw text message. 

# In[94]:


# Create a new column called Clean Message. 
df["Clean Message"] = df["SMS Message"].copy()
# View the first 5 lines to see the results.
df.head()


# #### <font color = '5D6D7E'><i>Removing spaces, line breaks, tabs, and numbers</i> </font> 

# Next, we replace all numbers with a space, remove spaces, line breaks, and makes, and make all words lowercase. We also remove all 1 letter words such as "r", and "u" (as in "how r u") because they also create a lot of noise in the dataset and may have a negative impact on results. 

# In[95]:


# Replace all numbers with a space and remove all extra spaces, line breaks, and tabs. 
df["Clean Message"] = df["Clean Message"].str.replace(r'\d+(\.\d+)?', '')
df["Clean Message"] = df["Clean Message"].str.replace(r'[^\w\d\s]', ' ')
df["Clean Message"] = df["Clean Message"].str.replace(r'\s+', ' ')
df["Clean Message"] = df["Clean Message"].str.replace(r'^\s+|\s+?$', '')

# Make all words lower case
df["Clean Message"] = df["Clean Message"].str.lower()

# Remove all 1 letter words
df["Clean Message"] = df["Clean Message"].str.replace(r'\b\w\b', '').str.replace(r'\s+', ' ')
df.head()


# #### <font color = '5D6D7E'><i>Removing punctuation, stopwords, and stemming</i> </font> 
# We next need to remove punctuation since, for example, "whats" and "what's" will not be considered the same word and will have a negative impact on the model by counting what should be considered the same word as two separate words. 
# 
# Stopwords, such as "I", "me", "you", "ours", etc., will also be removed because they do not provide us useful meaning to the data. They are also words that are used very often and will produce misleading results when, for instance, trying to determine the most commonly used words in spam messages. To remove stopwords, we use the NLTK library, which provides us with a list of commonly used stopwords to be removed from datasets. 
# 
# Stemming is also another important process that in implemented. Stemming, which is also provided in the NLTK library, is a process that translates all words into their root word. This makes words like "argue," "argued", argues" and "arguing" to be reduced to the stem, "argu", rather than counting each word as its own single word. 

# In[96]:


# The punctuation that will be removed from the data.
string.punctuation


# In[97]:


stopword = (stopwords.words('english')) # This gives you a list of words to be from the data set.
stopword[0:10] # See the first 10 stopwords


# In[98]:


# Create a function to remove all punctuation and stopwords.
stemmer = SnowballStemmer("english")
def textProcessing(message):  
    # Remove all punctuation and replace with a space
    message = message.translate(str.maketrans('','', string.punctuation))
    # Use Stemmer to translate words to ther root word
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    return " ".join(words)
df["Clean Message"] = df["Clean Message"].apply(textProcessing)


# Let's take a closer look at how the data has been cleaned. Notice how all punctuation, stop words, 1 letter words, etc. have been removed. We have also used stemmer so that words like "joking" and "joke" will be considered one word. 

# In[99]:


# Look at some random data to see how the messages have been cleaned. 
print("Raw Text Message #1:\t\t", df['SMS Message'][1])
print("Cleaned Message #1:\t\t", df['Clean Message'][1], '\n')

print("Raw Text Message #6:\t\t", df['SMS Message'][6])
print("Cleaned Message #6:\t\t", df['Clean Message'][6], '\n')

print("Raw Text Message #977:\t\t", df['SMS Message'][977])
print("Cleaned Message #977:\t\t", df['Clean Message'][977], '\n')

print("Raw Text Message #2:\t\t", df['SMS Message'][2])
print("Cleaned Message #2:\t\t", df['Clean Message'][2], '\n')

print("Raw Text Message #500:\t\t", df['SMS Message'][500])
print("Cleaned Message #500:\t\t", df['Clean Message'][500], '\n')

print("Raw Text Message #11:\t\t", df['SMS Message'][11])
print("Cleaned Message #11:\t\t", df['Clean Message'][11], '\n')

print("Raw Text Message #5567:\t\t", df['SMS Message'][5567])
print("Cleaned Message #5567:\t\t", df['Clean Message'][5567], '\n')


# ### <font color = 900C3F>Tokenization </font>
# This process returns only the words that are important by converting the string of text into a list of words (tokens). 

# In[100]:


# Create a list of tokens after messages have been cleaned
df['Tokens'] = df['Clean Message'].map(lambda text: nltk.tokenize.word_tokenize(text))


# In[101]:


# Create new column to see the length of messages after cleaning. 
df['Clean Length'] = df['Clean Message'].apply(len)
df.head()


# ### <font color = 900C3F>Visualizing the Data </font>

# Now that we've cleaned the data, the next thing we want to do is take a look at the words themselves to get an idea of the types of words used in spam vs. ham messages, which leads to the second research question. 

# ##### <font color="FF5733"> Research Question 2: Are there certain words that are more likely to show up in spam messages than non-spam messages? </font>

# We use Word Cloud to get a visualization of the top more frequently used words in ham and spam messages. The larger the word appears on the Word Cloud, the higher repitition the word occurs in our dataset. As we can see, the words "free," "text," and "call" appear to some of the top most frequently used words in spam messages.  

# In[102]:


# Create Word Cloud of the most frequently used words in spam messages
spam_msg = ' '.join(list(df[df['Class'] == "spam"]["Clean Message"]))
spam_count = WordCloud(width = 512, height = 512).generate(spam_msg)
plt.figure(figsize=(8,8), facecolor="k")
plt.imshow(spam_count)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# The words "go", "get", "ok" and "love" seem to be the most frequently used ham words. 

# In[103]:


# Create Word Cloud of the most frequently used words in ham messages
ham_msg = ' '.join(list(df[df['Class'] == "ham"]["Clean Message"]))
ham_count = WordCloud(width = 512, height = 512).generate(ham_msg)
plt.figure(figsize=(8,8), facecolor="k")
plt.imshow(ham_count)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# Let's look at the top 20 most common words in spam and non-spam messages. 

# In[104]:


ham_count = Counter(" ".join(df[df['Class']=='ham']["Clean Message"]).split()).most_common(20)
ham_count_df = pd.DataFrame.from_dict(ham_count)
ham_count_df = ham_count_df.rename(columns={0: "words in ham", 1 : "count"})

spam_count = Counter(" ".join(df[df['Class']=='spam']["Clean Message"]).split()).most_common(20)
spam_count_df = pd.DataFrame.from_dict(spam_count)
spam_count_df = spam_count_df.rename(columns={0: "words in spam", 1 : "count_"})


# In[105]:


print(ham_count_df[0:20])
print('\n',spam_count_df[0:20])


# Finally, we plot the top 20 ham and spam words using matplotlib. 

# In[106]:


ham_count_df.plot.bar(legend = False)
y_pos = np.arange(len(ham_count_df["words in ham"]))
plt.xticks(y_pos, ham_count_df["words in ham"])
plt.title('Most Frequent Words in Ham Messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

spam_count_df.plot.bar(legend = False)
y_pos = np.arange(len(spam_count_df["words in spam"]))
plt.xticks(y_pos, spam_count_df["words in spam"])
plt.title('Most Frequent Words in Spam Messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


# Our data shows that the words "Call" and "Free" are the top two most frequently used words in SMS spam messages. "Call" accounts for 15.4% of the top 20 words and "Free" accounts for 9.1% of the top 20 spam words.  

# In[107]:


explode = (0.1, 0.1, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
plt.pie(spam_count_df['count_'],
       labels = spam_count_df["words in spam"], 
       shadow = True,
       autopct='%1.1f%%', 
       explode = explode)
plt.title("Top 20 Words Used in Spam Messages")
plt.axis('equal')
plt.show()


# ## <font color = 581845>Machine Learning Models </font>

# So far we did exploratory data analysis of our project to undrestand our data. During this process we did sometransformations on our dataset and created some visualizations. Now we will actually get into our solution to predict if a message is ham or spam. 
# 
# As we can see our dataset is quite imbalanced. It contains about 86% ham messages and only 14 spam messages. If we run analysis on this model, even if we do random guessing, we will have accuracy of about 86%. To actually get better results from our dataset, we need to balance our dataset. 
# 
# There are several techniques for balancing the dataset that has imbalanced classes. Some of the methods we considered are following:
# 
# 1. Downsampling majority class
# 2. Upsamping minority class
# 3. Penalizing algorithms (Cost-Sensitive Training)
# 
# Out of these, we selected upsampliing minority class using resample method in sklearn library. We chose this becuase its easy to do and within the scope of this project.

# In[108]:


# Create a copy of the df dataframe. 
data = df.copy()
# Remove the unwanted columns.
data.drop(["Clean Message", "Length", "Tokens", "Clean Length", "Class"], axis = 1, inplace = True) 
# View the first few rows of the new dataframe. 
print(data.count())


# ### <font color = 900C3F>Creating Test and Train Data Sets </font>
# 
# Before upsampling the minority class, We will do a train and test split. We have to do this before resampling/upsampling becuase we do not want exact same observations in our training and testing datasets that could really bias our results.

# In[109]:


X_train, X_test, y_train, y_test = train_test_split(data['SMS Message'],
                                                    data['Label'],
                                                    test_size=0.3,
                                                    random_state=101)


# ### <font color = 900C3F>Balancing Dataset by Upsampling the Minority Class</font>

# In[110]:


# Concatinating X_train and y_train column to form a complete dataset having both SMS Message and Label columns
X_train_resample=pd.concat([X_train, y_train], axis=1)

all_ham=X_train_resample[X_train_resample['Label']==0] # All ham messages

all_spam=X_train_resample[X_train_resample['Label']==1] # All spam messages

#Upsampling the datset by using resample from sklearn library
spam_upsampled = resample(all_spam,replace=True,n_samples=3350,random_state=101) # bringing number of spam messages up to 4825 from 747

# Concatinating upsampled spam messages with ham messaegs to create a new and balanced dataset
df_upsampled = pd.concat([all_ham, spam_upsampled])

# Splitting into X_train and y_train again
X_train=df_upsampled['SMS Message']
y_train=df_upsampled['Label']


# In[111]:


# Create a bar chart to visualize the new upsampled dataset.

df_upsampled["Label"].value_counts().plot(kind = 'pie', 
                                          figsize = (6, 7))
                                          #legend=True)

plt.title('Ham vs. Spam Messages in Dataset')
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


# As we can see our dataset is now more balanced.

# ### Vectorization using TF-IDF Vectorizer
# 
# Once data have been split into training and testing datasets, we will go ahead and vectorize our training and testing data separately using very popular TF-IDF Vectorizer. 
# 
# Term Frequency - Inverse Document Frequency (TF-IDF), a popular term-weighting method, looks at the importance of a word in a corpus (a collection of documents). The TF-IDF approach creates a matrix from the messages, then calculates the frequency-inverse document frequency of a word. The more a word occurs in a document, the higher TF-IDF value. TF-IDF is also often "offset by the frequency of the word in the corpus," which allows it to understand how words such as "hey, bye, and, it, etc." are going to appear more frequently than other words. We are using this vectorizer becuase it will save us considerable amount of time as we will not have to create bag of words, and other steps using CountVectorizer().
# 
# Please note that we will not be vectorizing our entire dataset at once. This is to avoid information leakage from testing data to training data that could seriously bias our results.

# In[112]:


# Conveting the training data into TFIDF values.
tfidf = TfidfVectorizer(stop_words = 'english', use_idf = True, lowercase = True, strip_accents='ascii')
X_train_tfidf = tfidf.fit_transform(X_train) # Fit the data into a sparse matrix, then transform it
X_test_tfidf = tfidf.transform(X_test) # Frequency term matrix
print(X_train_tfidf.shape)


# The training data contains 6,700 rows and 6,983 columns. 

# In[113]:


X_train_tfidf


# We can see that dimensions of our training and testing data match. This means that we can feed it to our ML classifiers for predictions.

# In[118]:


print(X_train_tfidf.shape)
print(X_test_tfidf.shape)


# We use get_feature_names() to see some of the words that are in TF-IDF. 

# In[119]:


# Let's see a few words in TF-IDF
tfidf.get_feature_names()[1000:1020]


# To get an idea of how TF-IDF works, we look at some random lines in the X_train data set, which contains raw text (below) and see how they are transformed with TF-IDF. Below shows the raw text messages.  

# In[120]:


a = X_train_tfidf.toarray()
print(X_train.iloc[6],'\n')
print(X_train.iloc[600],'\n')
print(X_train.iloc[6000],'\n')


# Now we can see how the lines above were transformed after using TF-IDF. We can see that stopwords, punctuation, etc. that have been removed by using this method.

# In[121]:


# We can see that there are stopwords, punc that are removed when running it through tfidf
a = X_train_tfidf.toarray()
print(tfidf.inverse_transform(a[6]),'\n')
print(tfidf.inverse_transform(a[600]),'\n')
print(tfidf.inverse_transform(a[6000]),'\n')


# ## <font color = 581845>Experiments </font>
# ##### <font color="FF5733"> Research Question 3: Which ML algorithm can most accurately detect spam message?  </font>
# 
# Here we will test our vectorized data on some of the most commonly used ML Classifiers.
# 
# 1. Naive-Bayes Classifier
# 2. Support Vector Machines
# 3. Logistic Regression
# 
# In the end we will compare results of all three classifiers and suggest which model is best and why.

# ### <font color = 900C3F>Model 1: Naive Bayes MultinominalNB </font>

# The MultinominalNB Naive Bayes model will be used to train and predict spam messages. This method is good to use because it can train and predict spam messages at a very fast rate without the need to tune any parameters or scale any features. 

# In[122]:


# Training and predicting spam messages using Naive Bayes
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train_tfidf, y_train)


# In[123]:


print(len(X_test), len(y_test),'\n')
#print(y_test)[0]


# #### <font color = 'C70039'>Predictions </font>

# We can take a look at how NB is predicting the data. Below, is the first row in our test dataset (raw messages), which as a classification of 0. 

# In[124]:


NB_actual = np.array(y_test) #convert to a numpy array
print("Classification:\n", NB_actual[0])
print('')
print("Messages:")
print(X_test[0])


# The Naive Bayes classifier has also predicted that the first message is 0 (ham). 

# In[125]:


predictions = NB_classifier.predict(X_test_tfidf[0])
print(predictions)


# Now that we can see how the model works, we make predictions for the entire test dataset. 

# In[126]:


# Make predictions for the entire test dataframe. 
NB_predictions=NB_classifier.predict(X_test_tfidf)
#print(NB_predictions)


# The code below ensures that, whenever the prediction and the actual result are the same, count that number. This ensures that we get an accurate count of all of the correct predictions. 

# In[127]:


count = 0
for i in range (len(NB_predictions)):
    if NB_predictions[i] == NB_actual[i]:
        count = count + 1


# #### <font color = 'C70039'><i>Execution Times</i> </font>

# As previously stated, one of the advantages of using NB is that is works very fast. Below we can see the time it takes to predict, train, and classify our data. 

# In[128]:


import time
trainAndTestTime = {}
start = time.time()
NB_classifier.fit(X_train_tfidf, y_train)
NB_predictions=NB_classifier.predict(X_test_tfidf)
end = time.time()
trainAndTestTime["Multinomial_Naive_Bayes"] = end - start
print("{:7.7f} seconds".format(trainAndTestTime["Multinomial_Naive_Bayes"]))


# #### <font color = 'C70039'>Naive Bayes Model Evaluation </font>

# Using the Naive Bayes algorithm gives us an accuracy score of 97.18%. 

# In[150]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, NB_predictions)


# In[151]:


# We have 1,619 correct predictions
print("Correct predictions: ", count)
print("Total length of data: ", len(NB_predictions))
print("Accuracy: ", 1625/1672)


# There were 37 ham words that were predicted to be spam, and 10 spam words that were predicted as ham.

# In[152]:


m_confusion_test = metrics.confusion_matrix(y_test, NB_predictions)
pd.DataFrame(data = m_confusion_test, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam'])


# In[153]:


# Calculating accuracy score and displaying confusion matrix
print ("Accuracy Score:")
print("Accuracy:", accuracy_score(y_test, NB_predictions))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, NB_predictions))


# In[154]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, NB_predictions)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.title("Naive Bayes \n Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


# In[155]:


from sklearn.metrics import classification_report
print(classification_report(y_test, NB_predictions))


# ##### ROC and AUC for Naive Bayes

# In[201]:


probs=NB_classifier.predict_proba(X_test_tfidf)
roc_NB =  metrics.roc_curve(y_test,probs[:,1])

# fpr = False Positive Rate
# tpr = True Positive Rate
# threshholds is the pobability cutoff. In our case it is 0.5

fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])  
 
plot_roc_curve(fpr, tpr)
auc_nb= metrics.auc(fpr, tpr)
print('The Area Under the Curve is: ', auc_nb)


# There were no ham messages that were incorrectly classified as spam. 

# In[157]:


# print message text for the false positives (ham incorrectly classified as spam)
false_pos = X_test[NB_predictions > y_test]
print("# False Positive Messages: ", len(false_pos), '\n')
print("False Positive Messages:\n",false_pos)


# In[158]:


# print message text for the false negatives (spam incorrectly classified as ham)

false_neg = X_test[NB_predictions < y_test]
print("# False Negative Messages: ", (len(false_neg)),'\n')
print("False Negative Messages:\n",false_neg)


# In[159]:


# example false negative
X_test[3564]


# Of the 1,672 messages in the dataset, 1,475 messages are ham and 197 messages are spam. The classes are skewed since 88.21% of the data has a class of 0. This presents the problem of null accuracy, where one can achieve an accuracy score by always predicting the class that has the highest count, which in our case, is 0.

# In[160]:


# examine class distribution
print(y_test.value_counts())


# The null accuracy score is 88.21%, which means that a model that predicts all messages as 0 will have an accuracy score of 88.21%. This is a good way to set a minimum accuracy threshold our models should achieve. However, it also shows how models like the Naive Bayes model with a 97.18% accuracy score (less than 10% increased from null accuracy) still has room for improvement. This is where AUC and ROC are best to be used. 

# In[161]:


# Calculate the Null Accuracy Score
null_accuracy = max(y_test.mean(), 1-y_test.mean())
print("Null Accuracy Score: ", null_accuracy)


# In[162]:


probs=NB_classifier.predict_proba(X_test_tfidf)
auc_NB =  metrics.roc_curve(y_test,probs[:,1])


# ### <font color = 900C3F>Model 2: Support Vector Machines </font>

# The next algorithm we will use is SVM. Here we will use soft margin SVM that will provide better results. Soft margin SVM has two parameters to tune the model C value and the gamma. Instead of manually changing values and finding out the best parameters, we will use GridSearchCV functionaly in sklearn library to try a combination of different values. We will provide a range of values for C and gamma to the GridSearchCV and it will provide us the best parameters that will give us the best results.

# In[181]:


# Generating an array of C values we will be testing
c_values=np.arange(8,10,0.5)
# Generating an array of gamma values we will be testing
gamma_values=np.arange(0.1,1.1,0.1)

print('The C values are: ',c_values)
print('\n')
print('The gamma values are: ',gamma_values)


# In[182]:


param={'C':c_values, 'gamma':gamma_values}
grid_search = GridSearchCV(SVC(kernel='rbf'),param, cv=5)


# In[184]:


# takes about 30 minutes to run
grid_search.fit(X_train_tfidf,y_train)


# In[186]:


grid_search.best_params_


# #### <font color = 'C70039'>Predictions </font>

# In[187]:


SVM_predictions=grid_search.predict(X_test_tfidf)


# #### <font color = 'C70039'>SVM Model Evaluation </font>

# In[188]:


svm_confusion_mat = metrics.confusion_matrix(y_test, SVM_predictions)
pd.DataFrame(data = svm_confusion_mat, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam'])


# In[189]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, SVM_predictions)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.title("Support Vector Machine \n Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


# In[190]:


from sklearn.metrics import classification_report
print(classification_report(y_test,SVM_predictions))


# In[191]:


accuracy_score(y_test, SVM_predictions)


# ##### ROC and AUC for Support Vector Machines

# One of the disadvantages of SVM is that it doesnt directly provide probabilyt estimates. Therefore, in the interest of time, we will not calcualte ROC and AUC fo this project.

# ### <font color = 900C3F>Model 3: Logistic Regression </font>

# In[167]:


from sklearn.linear_model import LogisticRegression


# In[168]:


log_reg=LogisticRegression(random_state=101).fit(X_train_tfidf, y_train)


# #### <font color = 'C70039'>Predictions </font>

# In[169]:


log_predict=log_reg.predict(X_test_tfidf)


# In[170]:


log_confusion_mat = metrics.confusion_matrix(y_test, log_predict)
pd.DataFrame(data = log_confusion_mat, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam'])


# In[171]:


CM = confusion_matrix(y_test, log_predict)
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.title("Logistic Regression \nConfusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()


# #### <font color = 'C70039'>Logistic Requestion Model Evaluation </font>

# In[172]:


print(classification_report(y_test,log_predict))


# In[173]:


accuracy_score(y_test, log_predict)


# ##### ROC and AUC for Logistic Regression

# In[200]:


prob_log=log_reg.predict_proba(X_test_tfidf)
roc_logistic=metrics.roc_curve(y_test,prob_log[:,1])

fpr, tpr, thresholds = roc_curve(y_test, prob_log[:,1])
plot_roc_curve(fpr, tpr)
auc_log= metrics.auc(fpr, tpr)
print('The Area Under the Curve (AUC) is: ', auc_log)


# ## <font color = 581845>Results</font>
# 
# Here we can see that SVM has the hightest accuracy

# ##### Confusion Matrices

# In[198]:


print('Naive Bayes Confusion Matrix')
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam']))
print('\n')
print('SVM Confusion Matrix')
print(pd.DataFrame(data = svm_confusion_mat, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam']))
print('\n')
print('Logistic Regression Confusion Matrix')
print(pd.DataFrame(data = log_confusion_mat, columns = ['Predicted Ham', 'Predicted Spam'],
            index = ['Actual Ham', 'Actual Spam']))


# ##### Accuracies

# In[203]:


print('Naive Bayes Accuracy')
print(accuracy_score(y_test, NB_predictions))
print('\n')
print('SVM Accuracy')
print(accuracy_score(y_test, SVM_predictions))
print('\n')
print('Logistic Regression Accuracy')
print(accuracy_score(y_test, log_predict))


# ##### Classification Reports

# In[199]:


print('Naive Bayes Classification report')
print(classification_report(y_test, NB_predictions))
print('\n')
print('SVM Classification report')
print(classification_report(y_test,SVM_predictions))
print('\n')
print('Logistic Regression Classification report')
print(classification_report(y_test,log_predict))


# ##### AUC and ROC

# In[202]:


print('The Area Under the Curve for Naive Bayes')
print(auc_nb)
print('\n')
print('The Area Under the Curve for Logistic Regression')
print(auc_log)


# ## <font color = 581845>Analysis, and Conclusion </font>
# 
# In our analysis, we compared three supervised ML classifiers:
# 
# 1. Naive Bayes
# 2. Support Vector Machines (Soft Margin)
# 3. Logistic Regression
# 
# 
# According to our analysis, all three models perform extremely well. It is hard to prefer one over the other. However, we will chose a model that is easiest to implement and has higher interpretability. 
# 
# Our analysis shows that, while SVM had accuracy of 98.20% it had lower recall score for detecting spam messages. Naive Bayes has about 97% accuracy but if we look at false positive rates it is highest. However its AUC is very impressive which shows that is a very good classifier. Similarly, logistic regression has about 98% accracy and impressive AUC and recall scores. 
# 
# Based on the results, we will chose and recommend logistic regression. Other than good performance, the interpretability of logistic regression is higher than Naive Bays and SVM. Logistic regression is also very efficient. We will not choose SVM as it is very compute intensive and doesnt really provide us high interpretability.
# 
# 

# ## <font color = 581845>References </font>
# [1] Ahmed, I., Ali, R., Guan, D., Lee, Y., Lee, S., & Chung, T. (2015). Semi-supervised learning using frequent itemset and ensemble learning for SMS classification. Expert Systems with Applications, 42(3), 1065-1073. doi:10.1016/j.eswa.2014.08.054
# 
# [2] Barushka, A., & Hajek, P. (2018). Spam filtering using integrated distribution-based balancing approach and regularized deep neural networks. Applied Intelligence, 48(10), 3538-3556. doi:10.1007/s10489-018-1161-y
# 
# [3] Delany, S. J., Buckley, M., & Greene, D. (2012). SMS spam filtering: Methods and data. Expert Systems with Applications, 39(10), 9899-9908. doi:10.1016/j.eswa.2012.02.053
# 
# [4] Méndez, J. R., Cotos-Yañez, T. R., & Ruano-Ordás, D. (2019). A new semantic-based feature selection method for spam filtering. Applied Soft Computing Journal, 76, 89-104.doi:10.1016/j.asoc.2018.12.008
# 
# [5] Nagwani, N. K., & Sharaff, A. (2016). SMS spam filtering and thread identification using bi-level text classification and clustering techniques. Journal of Information Science, 43(1), 75-87. doi:10.1177/0165551515616310
# 
# [6] Navaney, P., Dubey, G., & Rana, A. (2018). SMS Spam Filtering Using Supervised Machine Learning Algorithms. 2018 8th International Conference on Cloud Computing, Data Science & Engineering (Confluence), 43-48. doi:10.1109/confluence.2018.8442564
# 
# [7] Razi, Z., & Asghari, S. A. (2017). Providing an Improved Feature Extraction Method for Spam Detection Based on Genetic Algorithm in an Immune System. Journal of Knowledge-Based Engineering and Innovation, 3(8), 596-605. doi:649123/10.112675
# 
# [8] Sajedi, H., Parast, G. Z., & Akbari, F. (2016). SMS Spam Filtering Using Machine Learning Techniques: A Survey. Machine Learning Research, 1(1), 1-14. doi:10.11648/j.mlr.20160101.11
# 
# [9] Sethi, P., Bhandari, V., & Kohli, B. (2017). SMS spam detection and comparison of various machine learning algorithms. 2017 International Conference on Computing and Communication Technologies for Smart Nation (IC3TSN), 28-31. doi:10.1109/ic3tsn.2017.8284445
# 
# [10] Sheu, J., Chu, K., Li, N., & Lee, C. (2017). An efficient incremental learning mechanism for tracking concept drift in spam filtering. Plos One, 12(2), 1-17. doi:10.1371/journal.pone.0171518
# 
# [11] Tripathy, A., Agrawal, A., & Rath, S. K. (2016). Classification of sentiment reviews using n-gram machine learning approach. Expert Systems with Applications, 57, 117-126. doi:10.1016/j.eswa.2016.03.028
# 
# [12] Uysal, A. K., Gunal, S., Ergin, S., & Gunal, E. S. (2013). The Impact of Feature Extraction and Selection on SMS Spam Filtering. Electronics and Electrical Engineering, 19(5), 67-72. doi:10.5755/j01.eee.19.5.1829

# <b>Additional Resources</b>
# 
# http://www.dt.fee.unicamp.br/~tiago//smsspamcollection/doceng11.pdf
# 
# https://www.kaggle.com/pablovargas/naive-bayes-svm-spam-filtering
# 
# http://inmachineswetrust.com/posts/sms-spam-filter/
# 
# https://www.kaggle.com/aditya094/nltk-stemming-countvectorizer-97-accuracy
# 
# https://github.com/surya-veer/SpamHam-Filter/blob/master/Spam_filter.ipynb
# 
# https://github.com/poojasastry/SMS_Classification/blob/master/Code%20and%20Result/MLProject_SMSClassification.ipynb
# 
# https://github.com/piskvorky/data_science_python/blob/master/data_science_python.ipynb
# 
# https://www.researchgate.net/publication/303097249_Text_Normalization_and_Semantic_Indexing_to_Enhance_Instant_Messaging_and_SMS_Spam_Filtering
# 
# https://github.com/justmarkham/DAT8/blob/master/other/model_comparison.md
# 
# http://sebastianraschka.com/Articles/2014_naive_bayes_1.html
# 
# https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
# 
# http://simranmetric.com/classifying-spam-sms-messages-using-naive-bayes-classifier/
# 
# http://inmachineswetrust.com/posts/sms-spam-filter/
# 
# https://www.ritchieng.com/machine-learning-multinomial-naive-bayes-vectorization/
# 
# https://github.com/poojasastry/SMS_Classification/blob/master/Code%20and%20Result/MLProject_SMSClassification.ipynb
