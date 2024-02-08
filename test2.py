import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

user_preferences = pd.Series([9.3, 1994,[1,2,3,4]], index=["IMDB_Rating", "Released_Year",'Test'])
user_preferences = pd.DataFrame([user_preferences])
print(user_preferences)


# Load the MovieLens dataset
data = pd.read_csv('test.csv')

# Assuming 'description' and 'actors' are columns containing movie descriptions and actors
# If your data doesn't have these columns, adjust accordingly
# You might consider other features like movie tags, genres, etc.

# Concatenate 'description' and 'actors' columns
data['combined_features'] = data['Overview'].fillna('') + ' ' + data['Star1'].fillna('')+ ' ' + data['Star2'].fillna('')+ ' ' + data['Star2'].fillna('')

# Use TfidfVectorizer for feature extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
movie_features_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])

# Feature extraction for the new movie
new_movie_title = 'The Dark Knight'
new_movie_description = 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one. Action, Crime, Drama'  # Replace with the actual description of the new movie
new_movie_actors = 'Christian Bale'  # Replace with the actual actors of the new movie
new_movie_combined_features = new_movie_description + ' ' + new_movie_actors
new_movie_vector = tfidf_vectorizer.transform([new_movie_combined_features])

# Compute cosine similarity between the new movie and existing movies
similarity_scores = cosine_similarity(new_movie_vector, movie_features_matrix)

# Get the indices of movies with high similarity scores
similar_movies_indices = similarity_scores.argsort()[0][::-1]

print(similar_movies_indices)
top_n = 10
top_recommendations = similar_movies_indices[:top_n]
for idx in top_recommendations:
    recommended_movie_title = data.loc[idx, 'Series_Title']
    print(f"Movie: {recommended_movie_title}, Similarity Score: {similarity_scores[0, idx]}")
