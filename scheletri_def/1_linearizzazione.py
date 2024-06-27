"""
Metodi di linearizzazione per la soluzione di sistemi non lineari
1. Bisezione
2. Regula falsi
3. Corde
4. Secanti
5. Newton
6. Newton modificato (velocità di convergenza c = 2 => quadratica)
7. Newton Raphson
Varianti di Newton Raphson
8. Newton con Shamanskii
9. Newton con corde
10. Newton per il minimo con Simpy (matrice Hessiana con la funzione)
11. Newton per il minimo senza Simpy (matrice Hessiana con array)
"""

import math 
import numpy as np
import numpy.linalg as npl

def sign(x): 
   return math.copysign(1, x)

def stima_ordine(xk,iterazioni):
     #Vedi dispensa allegata per la spiegazione

      k=iterazioni-4
      p=np.log(abs(xk[k+2]-xk[k+3])/abs(xk[k+1]-xk[k+2]))/np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1]))
     
      ordine=p
      return ordine

def metodo_bisezione(fname, a, b, tolx,tolf):
 """
 Implementa il metodo di bisezione per il calcolo degli zeri di un'equazione non lineare.
 """
 fa=fname(a)
 fb=fname(b)
 if sign(fa) * sign(fb) >= 0: #to do
    print("Non è possibile applicare il metodo di bisezione \n")
    return None, None,None

 it = 0
 v_xk = []
 maxit = math.ceil(math.log((b - a) / tolx) / math.log(2))-1
 
 while abs(b-a) >= tolx and it < maxit:  #to do
    xk = a + (b - a) / 2 #to do
    v_xk.append(xk)
    it += 1
    fxk=fname(xk)
    if fxk==0:
      return xk, it, v_xk
     
    if sign(fa)*sign(fxk)>0:   
       # to do
       a = xk
       fa = fxk
    elif sign(fxk)*sign(fb)>0:    
       # to do
       b = xk
       fb = fxk
 
 return xk, it, v_xk


def falsi(fname, a, b, maxit, tolx,tolf):
   """
    Implementa il metodo di falsa posizione per il calcolo degli zeri di un'equazione non lineare.
   """
   fa=fname(a)
   fb=fname(b)
   
   if sign(fa) * sign(fb) >= 0: #to do
    print("Non è possibile applicare il metodo di falsa posizione \n")
    return None, None,None
   
   it = 0
   v_xk = []
   fxk=10
   while it < maxit and abs(b - a) >= tolx and abs(fxk) >= tolf: # to do
        xk = a - fa * (b - a) / (fb - fa) #to do
        v_xk.append(xk)
        it += 1
        fxk=fname(xk)
        if fxk==0:
            return xk, it, v_xk
        
        if sign(fa)*sign(fxk)>0:   
            #to do
            a = xk       
            #to do
            fa = fxk
        elif sign(fxk)*sign(fb)>0:    
            #to do
            b = xk
            #to do
            fb = fxk
   return xk, it, v_xk


def corde(fname,m,x0,tolx,tolf,nmax):
    """
    Implementa il metodo delle corde per il calcolo degli zeri di un'equazione non lineare.
    """
    xk = []
    fx0 = fname(x0) #to do
    d = fx0 / m #to do
    x1 = x0 - d #to do
    fx1 = fname(x1)
    xk.append(x1)
    it=1
    
    while it < nmax and abs(d) >= tolx * abs(x1) and abs(fx1) >= tolf:
        x0 = x1 # to do
        fx0 = fname(x0) #to do
        d = fx0 / m #to do
        '''
        #x1= ascissa del punto di intersezione tra  la retta che passa per il punto
        (xi,f(xi)) e ha pendenza uguale a m  e l'asse x
        '''
        x1 = x0 - d #to do  
        fx1=fname(x1)
        it=it+1
        
        xk.append(x1)
        
    if it==nmax:
        print('raggiunto massimo numero di iterazioni \n')
    
    return x1,it,xk

def secanti(fname,xm1,x0,tolx,tolf,nmax):
    """
    Implementa il metodo delle secanti per il calcolo degli zeri di un'equazione non lineare.
    """
    xk=[]
    fxm1 = fname(xm1) #to do
    fx0 = fname(x0) #to do
    d = fx0 * (x0 - xm1) / (fx0 - fxm1) #to do
    x1 = x0 - d #to do
    xk.append(x1)
    fx1 =  fname(x1)
    it = 1
    
    while it<nmax and abs(fx1)>=tolf and abs(d)>=tolx*abs(x1):
        xm1 = x0 #to do
        x0 = x1 #to do
        fxm1 = fname(xm1) #to do
        fx0 = fname(x0) #to do 
        d = fx0 * (x0 - xm1) / (fx0 - fxm1) #to do
        x1 = x0 - d #to do
        fx1=fname(x1)
        xk.append(x1)
        it=it+1        
    
    if it==nmax:
        print('Secanti: raggiunto massimo numero di iterazioni \n')
    
    return x1,it,xk

def newton(fname,fpname,x0,tolx,tolf,nmax):
    xk=[]
    fx0 = fname(x0)
    if abs(fpname(x0)) <= np.spacing(1): #to do
        print(" derivata prima nulla in x0")
        return None, None,None
    
    d = fx0 / fpname(x0) #to do
    x1 = x0 - d #to do
    fx1=fname(x1)
    xk.append(x1)
    it=1
    
    while it < nmax and abs(d) >= tolx * abs(x1) and abs(fx1) >= tolf: #to do 
        x0 = x1 #to do
        fx0 = fname(x0)#to do
        if abs(fpname(x0)) <= np.spacing(1): #to do: #Se la derivata prima e' pià piccola della precisione di macchina stop
            print(" derivata prima nulla in x0")
            return None, None,None
        d = fx0 / fpname(x0) #to do
        x1 = x0 - d #to do
        fx1=fname(x1)
        it=it+1
        
        xk.append(x1)
        
    if it==nmax:
        print('raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk

def newton_mod(fname,fpname,m,x0,tolx,tolf,nmax):
    xk=[]
    fx0 = fname(x0) #to do
    if abs(fpname(x0)) <= np.spacing(1): #to do :
        print(" derivata prima nulla in x0")
        return None, None,None

    d = fx0 / (fpname(x0)) #to do
    x1 = x0 - m*d #to do
    fx1 = fname(x1) #to do
    xk.append(x1)
    it=1
    
    while it <nmax and abs(d) >= tolx * abs(x1) and abs(fx1) >= tolf:#to do :
        x0 = x1 #to do
        fx0 = fname(x0) #to do

        if abs(fpname(x0)) <= np.spacing(1): #to do: #Se la derivata prima e' pià piccola della precisione di macchina stop
            print(" derivata prima nulla in x0")
            return None, None,None
        
        d = fx0 / (fpname(x0)) # to do
        x1 = x0 - m*d #to do 
        fx1=fname(x1)
        it=it+1
        
        xk.append(x1)
        
    if it==nmax:
        print('raggiunto massimo numero di iterazioni \n')
        
    return x1,it,xk

def my_newtonSys(fun, jac, x0, tolx, tolf, nmax):
   """ 
   Newton Raphson
   """
   matjac = jac(x0)
   if npl.det(matjac) == 0: #to do:
    print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
    return None, None,None
   
   s = -npl.solve(matjac, fun(x0)) #to do
   # Aggiornamento della soluzione
   it = 1
   x1 = x0 + s #to do
   fx1 = fun(x1)
   Xm = [np.linalg.norm(s, 1)/np.linalg.norm(x1,1)]
   
   while it < nmax and npl.norm(fx1, 1) >= tolf and npl.norm(s, 1) >= tolx * npl.norm(x1, 1): #to do
        x0 = x1 #to do
        it += 1
        matjac = jac(x0)
        if npl.det(matjac) == 0: #to do:
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None

   s = -npl.solve(matjac, fun(x0)) #to do
   # Aggiornamento della soluzione
   x1 = x0 + s #to do
   fx1 = fun(x1)
   Xm.append(np.linalg.norm(s, 1)/np.linalg.norm(x1,1))
   
   return x1, it, Xm


def my_newtonSys_corde(fun, jac, x0, tolx, tolf, nmax):

  """
  Funzione per la risoluzione del sistema f(x)=0
  mediante il metodo di Newton, con variante delle corde, in cui lo Jacobiano non viene calcolato
  ad ogni iterazione, ma rimane fisso, calcolato nell'iterato iniziale x0.
  """
  matjac = jac(x0)   
  if npl.det(matjac) == 0: #to do:
    print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
    return None, None,None
  
  s = -npl.solve(matjac, fun(x0)) #to do
  # Aggiornamento della soluzione
  it = 1
  x1 = x0 + s #to do
  fx1 = fun(x1)
  Xm = [np.linalg.norm(s, 1)/np.linalg.norm(x1,1)]

  while it < nmax and npl.norm(s, 1) >= tolx * npl.norm(x1, 1) and npl.norm(fx1, 1) >= tolf: #to do:
    x0 = x1#to do
    it += 1  
    if npl.det(matjac) == 0: #to do:
        print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
        return None, None,None
    
    s = -npl.solve(matjac, fun(x0)) #to do

    # Aggiornamento della soluzione
    x1 = x0 + s #to do
    fx1 = fun(x1)
    Xm.append(np.linalg.norm(s, 1)/np.linalg.norm(x1,1))

  return x1, it, Xm

def my_newtonSys_sham(fun, jac, x0, tolx, tolf, nmax):

  """
  Funzione per la risoluzione del sistema f(x)=0
  mediante il metodo di Newton, con variante delle shamanski, in cui lo Jacobiano viene
  aggiornato ogni un tot di iterazioni, deciso dall'utente.
  """

  matjac = jac(x0)
  if npl.det(matjac) == 0: #to do:
    print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
    return None,None,None

  s = -npl.solve(matjac, fun(x0)) #to do
  # Aggiornamento della soluzione
  it = 1
  x1 = x0 + s #to do
  fx1 = fun(x1)

  Xm = [np.linalg.norm(s, 1)/np.linalg.norm(x1,1)]
  update=10  #Numero di iterazioni durante le quali non si aggiorna la valutazione dello Jacobiano nell'iterato attuale
  while it < nmax and npl.norm(s, 1) >= tolx* npl.norm(x1, 1) and npl.norm(fx1, 1) >= tolf: #to do:
    x0 = x1 #to do
    it += 1
    if it%update==0:   #Valuto la matrice di iterazione nel nuovo iterato ogni "update" iterazioni
        #to do
        matjac = jac(x0)
   
        if npl.det(matjac) == 0: #to do == 0:
           print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
           return None,None,None
        else:         
           s = -npl.solve(matjac, fun(x0)) #to do
    else:          
           s = -npl.solve(matjac, fun(x0)) #to do

    # Aggiornamento della soluzione
    x1 = x0 + s #to do
    fx1 = fun(x1)
    Xm.append(np.linalg.norm(s, 1)/np.linalg.norm(x1,1))

  return x1, it, Xm


def my_newton_minimo(gradiente, Hess, x0, tolx, tolf, nmax):

  """
  Funzione di newton-raphson per calcolare il minimo di una funzione in più variabili
  DA UTILIZZARE NEL CASO IN CUI CALCOLATE DRIVATE PARZIALI PER GRADIENTE ED HESSIANO SENZA UTILIZZO DI SYMPY
  """
  matHess = Hess(x0) #to do
  if npl.det(matHess) == 0: #to do:
    print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
    return None, None, None
  
  grad_fx0= gradiente(x0)    
  s = -npl.solve(matHess, grad_fx0) #to do
  # Aggiornamento della soluzione
  it = 1
  x1 = x0 + s #to do
  grad_fx1 = gradiente(x1)
  Xm = [np.linalg.norm(s, 1)]
  
  while it < nmax and npl.norm(s, 1) >= tolx * npl.norm(x1, 1) and npl.norm(grad_fx1,) >= tolf:#to do:
    x0 = x1 #to do
    it += 1
    matHess = Hess(x0) #to do
    grad_fx0=grad_fx1
     
    if npl.det(matHess) == 0: #to do:       
      print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
      return None, None, None
 
    s = -npl.solve(matHess, grad_fx0) #to do     
    # Aggiornamento della soluzione
    x1 = x0 + s#to do

    #Calcolo del gradiente nel nuovo iterato
    grad_fx1  = gradiente(x1)
    print(np.linalg.norm(s, 1))
    Xm.append(np.linalg.norm(s, 1))

  return x1, it, Xm

def my_newton_minimo_MOD(gradiente, Hess, x0, tolx, tolf, nmax):

  """
  Funzione di newton-raphson per calcolare il minimo di una funzione in più variabili, modificato nel caso in cui si utilizzando sympy 
  per calcolare Gradiente ed Hessiano. 
  Rispetto alla precedente versione cambia esclusivamente il modo di valutare il vettore gradiente e la matrice Hessiana in un punto 
  """    
  matHess = np.array([[Hess[0, 0](x0[0], x0[1]), Hess[0, 1](x0[0], x0[1])],
                      [Hess[1, 0](x0[0], x0[1]), Hess[1, 1](x0[0], x0[1])]])
 
  gradiente_x0=np.array([gradiente[0](x0[0], x0[1]),gradiente[1](x0[0], x0[1])])
   
  if npl.det(matHess) == 0:#to do
    print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
    return None, None, None
      
  s = -npl.solve(matHess, gradiente_x0) #to do
  
  # Aggiornamento della soluzione
  it = 1
  x1 = x0 + s #to do
  grad_fx1=np.array([gradiente[0](x1[0],x1[1]),gradiente[1](x1[0],x1[1])])
  Xm = [np.linalg.norm(s, 1)]
  
  while it < nmax and npl.norm(s, 1) >= tolx * npl.norm(x1, 1) and npl.norm(grad_fx1, 1) >= tolf: #to do:     
    x0 = x1
    it += 1
    #to do: 
    matHess = np.array([[Hess[0, 0](x0[0], x0[1]), Hess[0, 1](x0[0], x0[1])],
                      [Hess[1, 0](x0[0], x0[1]), Hess[1, 1](x0[0], x0[1])]])
    grad_fx0 = grad_fx1
      
    if np.linalg.det(matHess) == 0:       
      print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
      return None, None, None
        
    s = -npl.solve(matHess, grad_fx0) #to do
    # Aggiornamento della soluzione
    x1 = x0 + s #to do
    #Aggiorno il gradiente per la prossima iterazione 
    grad_fx1=np.array([gradiente[0](x1[0],x1[1]),gradiente[1](x1[0],x1[1])])
    print(np.linalg.norm(s, 1))
    Xm.append(np.linalg.norm(s, 1))

  return x1, it, Xm