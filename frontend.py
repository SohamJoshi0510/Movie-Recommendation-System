import streamlit as st 
import pickle
import heapq

movies = pickle.load(open("movies.pkl", "rb"))
movie_names = movies.title

similarity_matrix = pickle.load(open("similarity_matrix.pkl", "rb"))

def recommend(movie, n=10):
  index = movies[movies['title'] == movie].index[0]
  distances = similarity_matrix[index]
  movie_indices = heapq.nlargest(n+1, enumerate(distances), key=lambda x: x[1])
  movie_names = [movies.iloc[movie_index[0]].title for movie_index in movie_indices]

  return movie_names[1:], n

st.title("Movie Recommendation System")
selected_movie = st.selectbox("What's your go-to movie?", movie_names)


if st.button("Recommend"):
  recommendations, n = recommend(selected_movie)
  st.header("Here are a few movies that you might like.")
  for i in range(n):
    st.subheader(recommendations[i])

  st.balloons()


