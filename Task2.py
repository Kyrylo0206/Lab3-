import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

ratings_file_path = 'ratings.csv'
ratings_df = pd.read_csv(ratings_file_path)

ratings_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=50, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)
ratings_matrix_filled = ratings_matrix.fillna(ratings_matrix.mean().mean())
R = ratings_matrix_filled.values
user_ratings_mean = R.mean(axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U[:, 0], U[:, 1], U[:, 2])
ax.set_title('Users in Reduced 3D Space')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Vt[0, :], Vt[1, :], Vt[2, :])
ax.set_title('Movies in Reduced 3D Space')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
