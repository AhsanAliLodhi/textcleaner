# Textcleaner
## A Python module to clean/preprocess text, possibly for NLP tasks

One can use preprocess.py to preprocess a csv and generate clean data.

Example Use : preprocess --inputfile myfile.csv --textcolumn target

I've performed following preprocess steps in order.

1 - Try to decode text in correct encoding.
2 - Use of ` instead of ' as apostrophe was a common error in the text, we had to correct it.
3 - use a ML model to predict word spelling corrections such as "iam sad" -> "i am sad" or "hvae yuo eataen food" -> "Have you eaten food"
4- We use a smart model which uses google's word2vec to predict contractions such as "I've" to "I have"
5 - Then we ignore every other symbol except words
6 - We remove common stop words however, with an exception of negation words such as not, no and never. Idea was that while words like "The" and "So" do not help in our task of binary classification, negation words on the other hand while being common do make a lot of sense to keep because of their impact on meaning of a sentence.
7 - Lemmatize the words.

Please refer to Cleaner.py for more details.

### note: This project uses google's word2vec embeddings for smart contractions exapansion and this file is not optional at the moment. Also the embeddings must be stored as following GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin
