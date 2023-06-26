import nltk, string, spacy
from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt') # if necessary...
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=['abov', 'afterward', 'alon', 'alreadi', 'anywh', 'becau', 'el', 'elsewh', 'everywh', 'ind', 'otherwi', 'plea', 'somewh','alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 'cri', 'describ', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'formerli', 'forti', 'ha', 'henc', 'hereaft', 'herebi', 'hi', 'howev', 'hundr', 'inde', 'latterli', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'seriou', 'sever', 'sinc', 'sincer', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv', 'english','twenti', 'veri', 'wa', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv'])

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

f=open('/Users/rajakumarisureshbabu/HelloWorld/amiz/ftyu/kl.txt')
g=open('/Users/rajakumarisureshbabu/HelloWorld/amiz/ftyu/kk.txt')


AA=f.readlines()
BB=g.readlines()

a=' '
for i in AA:
    if list(set(i))[0]!=' ':
        i=i.rstrip('\n')
        i=i.lstrip()
        if len(i)>0:
            if i[0]!='#':
                a+=i+' '

b=' '
for i in BB:
    if list(set(i))[0]!=' ':
        i=i.rstrip('\n')
        i=i.lstrip()
        if len(i)>0:
            if i[0]!='#':
                b+=i+' '

print(cosine_sim(a,b)*100)

