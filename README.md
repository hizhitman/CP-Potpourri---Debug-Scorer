# CP-Potpourri---Debug-Scorer

Used cosine similarity between the contents of two text files containing the two code snippets. It uses the NLTK library for text processing, the scikit-learn library for TF-IDF vectorization, and the spaCy library for tokenization. 
This code was written as a part of the 'Debug' competition in CP Potpourri event, Shaastra.
Users are given a faulty code and are expected to make minimal changes to the given code to make it work. The code is parsed into a text file and all the comments are removed before they are assigned marks using the above similarity algorithm. 
