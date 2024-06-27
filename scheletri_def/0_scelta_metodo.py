from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import SolveTriangular

# dati = loadmat('test_06_05_2024.mat')

# A=dati["A"] 

# A=A.astype(float)

# b=dati["b"]

# b=b.astype(float)

A = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
b = np.sum(A, axis=1)

n,m = A.shape
print(A.shape,A)
nz = np.count_nonzero(A)/(n*m)
perc_nz = nz * 100
print("percentuale elementi diversi da zero", perc_nz)
plt.spy(A) #bianco dove gli elementi sono nulli, un puntino nero dove ci sono elementi 

# matrice di grande dimensioni --> da 400 x 4000 in su
# matrice di piccole dimensioni --> massimo 50 x 50
# matrice sparsa --> numero di elementi diversi da zero non supera il 33%
#cholesky è piu veloce di Gauss


m, n = A.shape
print(f"La matrice A é composta da {m} righe e {n} colonne")
if m == n:
    print("É una matrice quadrata")
else:
    print("É una matrice rettangolare")
    
if np.count_nonzero(A)*100/(m*n) < 33:
    print("La matrice é sparsa")
else:
    print("La matrice non é sparsa")
    
print(f"Condizionamento della matrice A {np.linalg.cond(A):.2f}")

print(f"Il rango della matrice é: {np.linalg.matrix_rank(A)}")

print(f"Il determinante della matrice é: {np.linalg.det(A)}")

if np.linalg.det(A) > 0 and A[0][0] > 0:
    print("La matrice é definita positiva")
else:
    print("La matrice non é definita positiva") 

if np.array_equal(A, A.T):
    print("La matrice é simmetrica")
else:
    print("La matrice non é simmetrica")