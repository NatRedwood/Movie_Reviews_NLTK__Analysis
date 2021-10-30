import nltk
from nltk.corpus import movie_reviews
import pandas as pd

len(nltk.corpus.movie_reviews.fileids())

movie_reviews.categories()

stats = []
for fileid in movie_reviews.fileids():
    num_chars = len(movie_reviews.raw(fileid))
    num_words = len(movie_reviews.words(fileid))
    num_sents = len(movie_reviews.sents(fileid))
    num_vocab = len(set(word.lower() for word in movie_reviews.words(fileid)))
    stats.append([fileid,round(num_chars/num_words),round(num_words/num_sents),round(num_words/num_vocab)])

df_stats = pd.DataFrame(stats, columns = ['Text ID', 'Average Word Length', 'Average Sentence Length', 'Linguistic Diversity'])
df_stats

cond_fd = nltk.ConditionalFreqDist(
    (sentiment, word)
    for sentiment in movie_reviews.categories()
    for word in movie_reviews.words(categories=sentiment))
sentiments = ['neg', 'pos']
target_words = ['good', 'bad', 'amazing', 'awful', 'no', 'not']
cond_fd.tabulate(samples = target_words, conditions = sentiments)

neg_words = movie_reviews.words(categories = 'neg')
pos_words = movie_reviews.words(categories = 'pos')
neg_words_count = len(neg_words)
pos_words_count = len(pos_words)
print(f"Total number of words in the positive review category: {pos_words_count}")
print(f"Total number of words in the negative review category: {neg_words_count}")

def pos_indicator(word='str'):
    word_pos_count = len([w for w in movie_reviews.words(categories = 'pos') if w == word])
    word_neg_count = len([w for w in movie_reviews.words(categories = 'neg') if w == word])
    count_total = word_neg_count + word_pos_count
    per_word = print(round(100*word_pos_count/count_total), "%")
    return per_word


pos_indicator('amazing')


pos_indicator('good')


def neg_indicator(word='str'):
    word_pos_count = len([w for w in movie_reviews.words(categories = 'pos') if w == word])
    word_neg_count = len([w for w in movie_reviews.words(categories = 'neg') if w == word])
    count_total = word_neg_count + word_pos_count
    per_word = print(round(100*word_neg_count/count_total), "%")
    return per_word

neg_indicator('bad')


neg_indicator('awful')

text_pos = nltk.Text(pos_words)
text_pos.concordance('not')


text_pos.concordance('no')

text_pos.concordance('bad')

text_neg = nltk.Text(neg_words)
text_neg.concordance('good')

print("Negative category:\n")
print(text_neg.common_contexts(['not','no']))
print("\nPositive category:\n")
print(text_pos.common_contexts(['not','no']))


print("Negative category:\n")
print(text_neg.common_contexts(['bad','awful']))
print("\nPositive category:\n")
print(text_pos.common_contexts(['bad','awful']))


print("Negative category:\n")
print(text_neg.common_contexts(['good','amazing']))
print("\nPositive category:\n")
print(text_pos.common_contexts(['good','amazing']))

text_neg.collocations()


text_pos.collocations()