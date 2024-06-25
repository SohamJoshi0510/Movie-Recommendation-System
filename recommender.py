
import numpy as np
import pandas as pd
import heapq

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')

movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# movies.isnull().sum()

movies.dropna(inplace=True)
# movies.isnull().sum()

import ast

movies.genres = movies.genres.apply(ast.literal_eval)
movies.genres = movies.genres.apply(lambda list_: [dict_['name'] for dict_ in list_])

movies.keywords = movies.keywords.apply(ast.literal_eval)
movies.keywords = movies.keywords.apply(lambda list_: [dict_['name'] for dict_ in list_])

movies.cast = movies.cast.apply(ast.literal_eval)
movies.cast = movies.cast.apply(lambda list_: [dict_['name'] for dict_ in list_][:3])


movies.crew = movies.crew.apply(ast.literal_eval)
movies.crew = movies.crew.apply(lambda list_: [dict_['name'] for dict_ in list_ if dict_['job'] == 'Director'])

movies.overview = movies.overview.apply(lambda str_: str_.split())

movies.genres = movies.genres.apply(lambda list_: [str_.replace(' ', '') for str_ in list_])
movies.keywords =  movies.keywords.apply(lambda list_: [str_.replace(' ', '') for str_ in list_])
movies.cast = movies.cast.apply(lambda list_: [str_.replace(' ', '') for str_ in list_])
movies.crew = movies.crew.apply(lambda list_: [str_.replace(' ', '') for str_ in list_])
# movies

movies['tags'] = movies.overview + movies.genres + movies.keywords + movies.cast + movies.crew
# movies

df_with_tags = movies[['id', 'title', 'tags']]
# df_with_tags

df_with_tags.loc[:, 'tags'] = df_with_tags.loc[:, 'tags'].apply(lambda list_: " ". join(list_))
df_with_tags.loc[:, 'tags'] = df_with_tags.loc[:, 'tags'].apply(lambda str_: str_.lower())
# df_with_tags

# Stemming
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(bag_of_words):
    list_of_words = bag_of_words.split()
    stemmed_list_of_words = [ps.stem(word) for word in list_of_words]
    return " ".join(stemmed_list_of_words)

df_with_tags.loc[:, 'tags'] = df_with_tags.loc[:, 'tags'].apply(stem)
# df_with_tags['tags']

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')

vectors = count_vectorizer.fit_transform(df_with_tags['tags']).toarray()
words = count_vectorizer.get_feature_names_out()
# for word in words:
#     print(word)
# vectors

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(vectors)
# similarity_matrix


import pickle
pickle.dump(similarity_matrix, open('similarity_matrix.pkl', 'wb'))


# if __name__ == '__main__':
#   print(recommend('Batman Begins'))
