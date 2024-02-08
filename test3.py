import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the MovieLens dataset
data = pd.read_csv('test.csv')

# Map user and movie IDs to continuous indices
user_mapping = {user_id: idx for idx, user_id in enumerate(data['userId'].unique())}
movie_mapping = {movie_id: idx for idx, movie_id in enumerate(data['movieId'].unique())}

data['user_idx'] = data['userId'].map(user_mapping)
data['movie_idx'] = data['movieId'].map(movie_mapping)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# Define the PyTorch dataset and dataloaders
class MovieLensDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe[['user_idx', 'movie_idx']].values
        self.labels = dataframe['rating'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_movie_pair = torch.LongTensor(self.data[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return {'user_movie_pair': user_movie_pair, 'label': label}


train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the matrix factorization model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size=30):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_size)

    def forward(self, user_movie_pair):
        user_idx, movie_idx = user_movie_pair[:, 0], user_movie_pair[:, 1]
        user_embedding = self.user_embeddings(user_idx)
        movie_embedding = self.movie_embeddings(movie_idx)
        prediction = (user_embedding * movie_embedding).sum(dim=1, keepdim=True)
        return prediction


# Instantiate the model, loss function, and optimizer
num_users = len(user_mapping)
num_movies = len(movie_mapping)
model = MatrixFactorization(num_users, num_movies, embedding_size=30)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        user_movie_pair, label = batch['user_movie_pair'], batch['label']
        prediction = model(user_movie_pair)
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            user_movie_pair, label = batch['user_movie_pair'], batch['label']
            prediction = model(user_movie_pair)
            loss = criterion(prediction, label)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_loss:.4f}')

# Making predictions for a specific user
user_id_to_predict = 1
user_idx_to_predict = user_mapping[user_id_to_predict]
all_movie_indices = torch.arange(num_movies)
user_movie_pairs_to_predict = torch.cat(
    [user_idx_to_predict * torch.ones_like(all_movie_indices).unsqueeze(1), all_movie_indices.unsqueeze(1)], dim=1)
predictions = model(user_movie_pairs_to_predict).squeeze().detach().numpy()

# Get top N recommendations
top_n = 10
top_indices = predictions.argsort()[-top_n:][::-1]
top_movie_ids = [movie_id for movie_id, idx in movie_mapping.items() if idx in top_indices]

# Display the top N recommendations
print(f"Top {top_n} recommendations for user {user_id_to_predict}:")
for movie_id in top_movie_ids:
    movie_title = data[data['movieId'] == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}")
