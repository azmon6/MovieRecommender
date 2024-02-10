
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


csv_file = 'movies_test.csv'
df2 = pd.read_csv(csv_file, sep=';')
print(df2)
# Specify the path to your CSV file
# csv_file_path = 'imdb_top_1000.csv'
#
# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv(csv_file_path)
#
# # Display the DataFrame
# df.drop(df.columns[0], axis=1, inplace=True)
# df.to_csv("test.csv", sep=',',index=False)

csv_file_path2 = 'test.csv'
df2 = pd.read_csv(csv_file_path2)

print(df2.columns)

df2['Actors'] = df2['Star1'] + " " + df2['Star2'] + " " + df2['Star3'] + " " + df2['Star4']
df2.rename(columns={'Star1':"Actors"})

csv_file_path3 = "smallDataset.csv"
df3 = df2[['Series_Title','IMDB_Rating','Released_Year','Overview','Genre','Actors']]
df3.to_csv("smallDataset.csv", sep=',',index=False)
exit(0)
combined_texts = [f"{df2['Series_Title'][ind]} {df2['Overview'][ind]} {df2['Genre'][ind]}" for ind in df2.index]

ratings = [df2['Meta_score'][ind] for ind in df2.index]
actors = [
    [df2['Star1'][ind], df2['Star2'][ind], df2['Star3'][ind]] for ind in df2.index
]
actor_strings = [' '.join(actor) for actor in actors]

text_vectorizer = TfidfVectorizer()
vectorized_texts = text_vectorizer.fit_transform(combined_texts)

# scaler = StandardScaler()
# scaled_ratings = scaler.fit_transform([[rating] for rating in ratings])

# Combine text and numerical features
feature_matrix = hstack([vectorized_texts, text_vectorizer.transform(actor_strings)])
# feature_matrix = hstack([vectorized_texts, scaled_ratings, text_vectorizer.transform(actor_strings)])

print(feature_matrix.toarray())

user_combine_text = "Dark knight. Christian Bale. Drama Action"
user_actors = ["Christian Bale"]
new_movie_vector = text_vectorizer.fit_transform([user_combine_text])

print(new_movie_vector)
similarity_scores = cosine_similarity(new_movie_vector, feature_matrix)
#
# similar_movies_indices = similarity_scores.argsort()[0][::-1]
# print(similar_movies_indices)

# Prompt -> extract keywords (RAKE + POS tagging -NLTK) -> feed into RandomForest (SKlearn) -> get answer from user - > Senitment Analysis(NLTK) -> redo if sadge