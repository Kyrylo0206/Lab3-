import numpy as np

def compute_svd(A):
    AtA = np.dot(A.T, A)
    AAt = np.dot(A, A.T)

    eigenvalues_AtA, V = np.linalg.eigh(AtA)

    idx = np.argsort(eigenvalues_AtA)[::-1]
    eigenvalues_AtA = eigenvalues_AtA[idx]
    V = V[:, idx]

    eigenvalues_AAt, U = np.linalg.eigh(AAt)

    idx = np.argsort(eigenvalues_AAt)[::-1]
    eigenvalues_AAt = eigenvalues_AAt[idx]
    U = U[:, idx]

    singular_values = np.sqrt(eigenvalues_AtA)

    Sigma = np.zeros_like(A, dtype=float)
    np.fill_diagonal(Sigma, singular_values)

    singular_values = np.abs(singular_values)

    A_reconstructed = np.dot(U, np.dot(Sigma, V.T))

    return U, Sigma, V.T, A_reconstructed

A = np.array([[1, 2, 3], [4, 5, 6]])
U, Sigma, Vt, A_reconstructed = compute_svd(A)

np.set_printoptions(precision=3, suppress=True)

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

print("______________________________________________________________________________________")
