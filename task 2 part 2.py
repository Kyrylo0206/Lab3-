import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

min_user_ratings = 10
min_movie_ratings = 20
filtered_ratings = ratings.groupby('userId').filter(lambda x: len(x) >= min_user_ratings)
filtered_ratings = filtered_ratings.groupby('movieId').filter(lambda x: len(x) >= min_movie_ratings)

ratings_matrix = filtered_ratings.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=50, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)

ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

normalized_ratings_sparse = csr_matrix(R_demeaned)

k = 50
U, sigma, Vt = svds(normalized_ratings_sparse, k=k)
sigma = np.diag(sigma)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:, 0], U[:, 1], U[:, 2])

ax.set_title('Users in Reduced Dimensional Space')
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Vt.T[:, 0], Vt.T[:, 1], Vt.T[:, 2])

ax.set_title('Movies in Reduced Dimensional Space')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
ax.set_zlabel('V3')
plt.show()

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)


def recommend_movies(predictions_df, user_id, movies_df, original_ratings_df, num_recommendations=10):
    user_row_number = user_id - 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    user_data = original_ratings_df[original_ratings_df.userId == user_id]
    user_full = (user_data.merge(movies_df, how='left', left_on='movieId', right_on='movieId').
                 sort_values(['rating'], ascending=False))

    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                       assign(PredictedRating=sorted_user_predictions).
                       sort_values('PredictedRating', ascending=False).
                       head(num_recommendations))

    return user_full, recommendations


user_id = 1
already_rated, predictions = recommend_movies(preds_df, user_id, movies, ratings, 10)

print("User {} has already rated:".format(user_id))
print(already_rated[['title', 'genres', 'rating']])

print("\nTop 10 movie recommendations for user {}:".format(user_id))
print(predictions[['title', 'genres', 'PredictedRating']])

for user_id in [2, 3, 4]:
    _, user_predictions = recommend_movies(preds_df, user_id, movies, ratings, 10)
    print("\nTop 10 movie recommendations for user {}:".format(user_id))
    print(user_predictions[['title', 'genres', 'PredictedRating']])
