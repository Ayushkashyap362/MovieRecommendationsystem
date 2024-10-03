import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

movies=pd.read_csv('movies.csv')
movies.head()

#Selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']

#Replacing the null values with null string
for feature in selected_features:
    movies[feature] = movies[feature].fillna('')

list_of_titles = movies['title'].tolist()
j = 0
for i in list_of_titles:
    j+=1
    print(j,i)

combined_features = movies['genres']+' '+movies['keywords']+' '+movies['tagline']+' '+movies['cast']+' '+movies['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#Getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

#Enter the movie name to get Similarity 
movie_name = input('Enter the movie to get similar movie suggestions : ')

find_close_match = difflib.get_close_matches(movie_name, list_of_titles)
close_match = find_close_match[0]

#Finding the index of the movie with the title
index_movie = movies[movies.title == close_match]['index'].values[0]
 
similarity_score = list(enumerate(similarity[index_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies[movies.index==index]['title'].values[0]
    if (i<30):
        print(i, '.',title_from_index)
        i+=1

movie_name = input('Enter your favourite movie name : ')
list_of_titles = movies['title'].tolist()
find_close_match = difflib.get_close_matches(movie_name, list_of_titles)
close_match = find_close_match[0]
index_movie = movies[movies.title == close_match]['index'].values[0]
similarity_score = list(enumerate(similarity[index_movie]))
sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print('Movies suggested for you with respect to movie name: \n')
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies[movies.index==index]['title'].values[0]
    if (i<30):
        print(i, '.',title_from_index)
        i+=1
