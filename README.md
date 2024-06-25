# Movie-Recommendation-System
This repository contains a Movie Recommendation System that suggests movies based on the similarity of their content. The project includes a backend script for processing and analyzing movie data, and a frontend application for user interaction.

# Dataset Link
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Features

- **Content-Based Filtering**: Uses movie metadata such as genres, keywords, cast, and crew to generate movie recommendations.  
- **Cosine Similarity**: Calculates the similarity between movies using cosine similarity of their feature vectors.  
- **Stemming and Vectorization**: Applies text preprocessing techniques like stemming and vectorization to enhance the recommendation accuracy.  
- **Streamlit Interface**: Provides a simple and interactive UI for users to get movie recommendations.

## Repository Structure

- **recommender.py**: Backend script that processes movie data, generates feature vectors, and calculates similarity scores.  
- **frontend.py**: Streamlit application that provides a user interface to get movie recommendations.

## Getting Started

### Prerequisites

- Python 3.x  
- Required Python libraries: `numpy`, `pandas`, `nltk`, `scikit-learn`, `streamlit`, `pickle`
  
## How It Works

### Data Processing:
- Merges movie and credits datasets.
- Extracts and preprocesses relevant features (genres, keywords, cast, crew).
- Creates a combined feature (tags) for each movie.

### Feature Vectorization:
- Applies stemming to tags.
- Uses CountVectorizer to convert text data into feature vectors.

### Similarity Calculation:
- Calculates cosine similarity between movie vectors.
- Saves the similarity matrix for future use.

### Recommendation:
- Retrieves the top N most similar movies based on cosine similarity.

### Usage
- Select a movie from the dropdown list.
- Click the "Recommend" button.
- View the list of recommended movies.

### Acknowledgements
- Data sourced from TMDB.
- Inspired by various tutorials and articles on movie recommendation systems.
