import pandas as pd
import jieba
import numpy
import jieba.analyse
#主题,文章,词
from gensim import corpora, models, similarities
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import pylab



df_news = pd.read_table('./val.txt', names = ['category', 'theme', 'URL', 'content'], encoding = 'utf-8')
df_news = df_news.dropna()


content = df_news.content.values.tolist()

content_S = []
for line in content:
    current_segment = jieba.lcut(line)
    #成功分到词并且不包含换行符之类的
    if len(current_segment) > 1 and current_segment != '\r\n' :
        content_S.append(current_segment)


df_content = pd.DataFrame({'content_S' : content_S})
#print(df_content.head())

stopwords = pd.read_csv('stopwords.txt', index_col = False, sep = '\t',
                        quoting = 3, names=['stopword'], encoding = 'utf-8')
#print(stopwords.head())

def drop_stopwords(contents, stopwords):

    content_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        content_clean.append(line_clean)

    return content_clean, all_words

contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
content_clean, all_words = drop_stopwords(contents, stopwords)


df_content = pd.DataFrame({'contents_clean': content_clean})
#print(df_content.head())

df_all_words = pd.DataFrame({'all_words' : all_words})
#print(df_all_words.head())

words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({'count' : numpy.size})
words_count = words_count.reset_index().sort_values(by = ['count'], ascending=False)
print(words_count.head())



'''
    画个画
'''

matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
wordcloud = WordCloud(font_path='./simhei.ttf', background_color='white', max_font_size=80)
word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
pylab.show()


#index = 1000
#content_S_str = ''.join(content_S[index])
#print(' '.join(jieba.analyse.extract_tags(content_S_str, topK=7, withWeight=False)))


#做映射,相当于词袋
dictionary = corpora.Dictionary(content_clean)

print(dictionary)

corpus = [dictionary.doc2bow(sentence) for sentence in content_clean]

print('====')
#print(corpus)

#语料库  映射字典  指定主题个数
lda = gensim.models.LdaModel(corpus = corpus, id2word=dictionary, num_topics = 20)
#print(lda.print_topic(3, topn=5))

for topic in lda.print_topics(num_topics=20, num_words=5):
    print(topic[1])


df_train = pd.DataFrame({'content_clean':content_clean, 'label':df_news['category']})
#print(df_train.tail())

#print(df_train.label.unique())

label_mapping = {'汽车':1, '财经':2, '科技':3, '健康':4, '体育':5, '教育':6, '文化':7, '军事':8, '娱乐':9, '时尚':10}
df_train['label'] = df_train['label'].map(label_mapping)

x_train, x_test, y_train, y_test = train_test_split(df_train['content_clean'].values, df_train['label'].values)

words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index)

#将文本转换为每个词出现的个数的向量
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
vec.fit(words)

#贝叶斯
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)

print(classifier)

test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[line_index]))
    except:
        print(line_index)

print(classifier.score(vec.transform(test_words), y_test))
#classifier.score(vec.transform(test_words), y_test)



'''
vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=False)
vectorizer.fit(words)
'''
