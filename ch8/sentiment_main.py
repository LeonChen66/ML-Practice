# -*- coding: utf-8 -*-
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
df = pd.read_csv('./movie_data.csv',encoding='cp1252')
# print(df.head(3))

# count = CountVectorizer()
# docs = np.array(['The sun is shining',
#                  'The weather is sweet'
#                  ,'The sun is shining and the weather is sweet'])
# bag = count.fit_transform(docs)
# print(count.vocabulary_)
# print(bag.toarray())
#
# tfidf = TfidfTransformer()
# np.set_printoptions(precision=2)
# print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
# print(df.loc[0, 'review'][-50:])
#
def preprocessor(text):
    text = re.sub('<[^>]*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower()) + ''.join(emotions).replace('-','')
    return text

# print(preprocessor(df.loc[0, 'review'][-50:]))
# print(preprocessor("</a>This :) is :( a test :-)!"))
# df['review'] = df['review'].apply(preprocessor)
#
def tokenizer(text):
    return text.split()

porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
#
# print(tokenizer('runners like running and thus they run'))
# print(tokenizer_porter('runners like running and thus they run'))

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

if __name__=="__main__":
    nltk.download('stopwords')
    stop = stopwords.words('english')
    # [w for w in tokenizer_porter('a runner likes running and runs a lot')]

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    # tfidf = tfidf.fit(X_train)
    param_grid = [{'vect__ngram_range':[(1,1)],'vect__stop_words':[stop, None],'vect__tokenizer':[tokenizer,tokenizer_porter],
                   'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]},
                  {'vect__ngram_range':[(1,1)],'vect__stop_words':[stop, None],'vect__tokenizer':[tokenizer,tokenizer_porter],
                   'vect__use_idf':[False],'vect__norm':[None],'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]}]

    lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    gs_lr_tfidf.fit(X_train,y_train)

    print('Best parameter set: %s' %gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' %gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' %clf.score(X_test, y_test))
    #
# def tokenizer(text):
#     text = re.sub('<[^>]*>', '', text)
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P')', text