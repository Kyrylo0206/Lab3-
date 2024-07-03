import numpy as np

def compute_svd(A):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigenvalues_AtA, V = np.linalg.eigh(AtA)
    eigenvalues_AAt, U = np.linalg.eigh(AAt)

    idx = np.argsort(eigenvalues_AtA)[::-1]
    V = V[:, idx]
    eigenvalues_AtA = eigenvalues_AtA[idx]

    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, np.sqrt(eigenvalues_AtA))

    A_reconstructed = np.dot(U, np.dot(Sigma, V.T))

    return U, Sigma, V.T, A_reconstructed

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, Sigma, Vt, A_reconstructed = compute_svd(A)

print("Матриця U:")
print(U)
print("\nМатриця Σ:")
print(Sigma)
print("\nМатриця V^T:")
print(Vt)
print("\nВідновлена матриця A:")
print(A_reconstructed)

print("\nЧи близька відновлена матриця до початкової:", np.allclose(A, A_reconstructed))

reconstructed_check = np.dot(U, np.dot(Sigma, Vt))
print("\nВідновлена матриця шляхом множення U, Σ та V^T:")
print(reconstructed_check)
print("\nЧи близька відновлена матриця шляхом множення до початкової:", np.allclose(A, reconstructed_check))
