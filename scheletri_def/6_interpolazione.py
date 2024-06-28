import numpy as np

def plagr(xnodi,j):
    """
    Restituisce i coefficienti del j-esimo polinomio di Lagrange
    associato ai punti del vettore xnodi. 
    Si utilizza il polinomio di Lagrange perché la matrice di Vandermonde è molto mal condizionata.
    """
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri=xnodi[1:n]
    else:
       xzeri=np.append(xnodi[0:j], xnodi[j+1, n]) #to do: polinomio i cui zeri sono gli elementi xzeri
    
    num = np.poly(xzeri) #to do
    den = np.polyval(num, xnodi[j]) #to do: valore del numeratore calcolato nel nodo xj.    
    p=num/den
    
    return p

def InterpL(x, y, xx):
     """
      funzione che determina in un insieme di punti il valore del polinomio
      interpolante ottenuto dalla formula di Lagrange.
     """
     n=x.size
     m=xx.size
     L=np.zeros((m,n))
     for j in range(n):
        p = plagr(x, j) #to do
        L[:,j] = np.polyval(p, xx) #to do: valuta il polinomio di Lagrange nei punti xx
    
     return L@y