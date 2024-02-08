from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


# Sample data for multiple movies
movies_data = [
    {"title": "The Amazing Adventure", "description": "A thrilling journey of discovery and excitement.", "genres": ["Action", "Adventure", "Thriller"], "rating": 8.5, "main_actor": "John Smith", "second_actor": "Jane Doe", "third_actor": "Sam Brown"},
    {"title": "Mystery Mansion", "description": "An eerie tale set in a haunted mansion.", "genres": ["Mystery", "Horror"], "rating": 7.2, "main_actor": "Alice Johnson", "second_actor": "Bob Williams", "third_actor": "Chris Davis"},
    # Add more movie data as needed
]

# Combine title, description, and genres for each movie into a single text
combined_texts = [f"{movie['title']} {movie['description']} {' '.join(movie['genres'])}" for movie in movies_data]

# Extract numerical features
ratings = [movie['rating'] for movie in movies_data]
actors = [
    [movie['main_actor'], movie['second_actor'], movie['third_actor']] for movie in movies_data
]

# Convert actor names to a single string for each movie
actor_strings = [' '.join(actor) for actor in actors]

# Create a TfidfVectorizer for text data
text_vectorizer = TfidfVectorizer()
vectorized_texts = text_vectorizer.fit_transform(combined_texts)

# Standardize numerical features
scaler = StandardScaler()
scaled_ratings = scaler.fit_transform([[rating] for rating in ratings])

# Combine text and numerical features
feature_matrix = hstack([vectorized_texts, scaled_ratings, text_vectorizer.transform(actor_strings)])

print(feature_matrix.toarray())