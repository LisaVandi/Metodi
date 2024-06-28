import numpy as np
import scipy.linalg as spLin
import SolveTriangular

"""
Un sistema lineare si dice sovradeterminato se #righe > #colonne. 
La sua risoluzione rappresenta un problema mal posto, in quanto potrebbe non avere soluzione o potrebbe essere non unica. 
"""

def eqnorm(A,b):
    """
    Risolve un sistema sovradeterminato con il metodo delle equazioni normali
    
    Condizioni necessarie: 
        - A deve avere rango massimo
        - A.T @ A deve essere simmetrica e definita positiva 
    
    Si cerca il vettore x* che rende minima la norma 2 al quadrato del residuo.
    Le 2 equazioni normali utilizzate (sapendo G = A.T @ A) sono: 
        Gradiente = 2Gx - 2 * A.T@b = 0
        Gx = A.T@b

    Il vettore x* si può trovare come soluzione del sistema A.T @ A @ x = A.T @ b 
    attraverso il metodo di Cholesky.

    NB: Se A è una matrice ben condizionata, questo metodo è ottimo. 
    Se A è mediamenente mal condizionata, questo metodo è numericamente pericoloso.
    """
    G = A.T @ A # to do   
    f = A.T @ b # to do

    L = spLin.cholesky(G, lower=True) # to do: per metodo di Cholesky
    U = L.T
     
    z = SolveTriangular.Lsolve(L, f) # to do
    x = SolveTriangular.Usolve(U, x) # to do
    
    return x
    
def qrLS(A,b):
    """
    Risolve un sistema sovradeterminato con il metodo QR-LS. 
    Q matrice ortogonale, R triangolare superiore. 
    Si risolve il sistema R@x = h, dove h = Q.T @ b => x = Usolve(R[0:n], h[0:n])

    Condizione necessaria: A di rango massimo

    Rispetto a LS, lavora solo sulla matrice A (e non su A@A.T che è peggio condizionata)
    """
    n=A.shape[1]  # numero di colonne di A
    Q,R=spLin.qr(A)

    h = Q.T @ b #to do
    x,flag=SolveTriangular.Usolve(R[0:n, 0:n], h[0:n]) #to do

    residuo=np.linalg.norm(h[n:])**2
    return x,residuo

def SVDLS(A,b):
    """
    Decomposizione in valori singolari per la soluzione del problema dei singoli quadrati.

    Vale: 
        - d = U.T @ b  
        - d1 = d [0:k] 
        - ci = di / sigmai

    ATTENZIONE! La matrice A non è a rango massimo ed è altamente mal condizionata
    """
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=spLin.svd(A)  #Attenzione : Restituisce U, il numpy-array 1d che contiene la diagonale della matrice Sigma e VT=VTrasposta)
    
    # Quindi 
    V=VT.T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=np.count_nonzero(s>thresh)
    print("rango=",k)

    d = U.T @ b #to do
    d1 = d[0:k] #to do: primi k elementi della diagonale d
    s1 = np.max([s[0:k]]) #to do
    # Risolve il sistema diagonale di dimensione kxk avene come matrice dei coefficienti la matrice Sigma
    c = d1 / s1 #to do

    x=V[:,:k]@c 
    residuo=np.linalg.norm(d[k:])**2
    return x,residuo
