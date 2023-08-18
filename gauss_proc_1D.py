from numpy.linalg import inv

import matplotlib.pyplot as plt
import numpy as np

X = np.array([1., 0., 5., -1., 7.,-5.,18.,3.,-13., 9.])

Y = X * np.sin(X)

X = X[:,np.newaxis]  #transformer X en colonne


              #Exemple avec des données 1D

sigma_n = 1.5

plt.grid(True,linestyle='--')

plt.errorbar(X, Y, yerr=sigma_n, fmt='o')

plt.title('Nuage de point', fontsize=7)

plt.xlabel('x')
plt.ylabel('y')

#plt.savefig('gaussian_processes_1d_fig_01.png', bbox_inches='tight')

             #Calculer la matrice de covariance K
sigma_f = 10.0

l = 1.0

X_dim1 = X.shape[0] #dim de X

D = np.zeros((X_dim1,X_dim1)) #matrice nulle 6x6 

K = np.zeros((X_dim1,X_dim1)) #matrice nulle 6x6 


D = X - X.T # X-X'

K = sigma_f**2*np.exp((-D*D)/(2.0*l**2)) #matrice de covariance

np.fill_diagonal(K, K.diagonal() +sigma_n**2 )

             #Faire une prediction en 1 point donnée
x_new = 2.0 # 2.5

D_new = np.zeros((X_dim1))

K_new = np.zeros((X_dim1))

D_new = X - x_new

K_new = sigma_f**2*np.exp((-D_new*D_new)/(2.0*l**2))

K_inv = inv(K)

m1 = np.dot(K_new[:,0],K_inv)

y_predict = np.dot(m1,Y)

print(y_predict)

             #Faire une prediction sur tout le graphe
X_new = np.linspace(-15,20,100)

Y_predict = []

Y_VAR_predict = []

plt.errorbar(X, Y, yerr=sigma_n, fmt='o') #Vrai graphe
#graphe de prediction
for x_new in X_new:

    D_new = np.zeros((X_dim1))

    K_new = np.zeros((X_dim1))

    D_new = X - x_new

    K_new = sigma_f**2*np.exp((-D_new*D_new)/(2.0*l**2))

    m1 = np.dot(K_new[:,0],K_inv)

    y_predict = np.dot(m1,Y)

    Y_predict.append(y_predict)

    y_var_predict = K[0,0] - K_new[:,0].dot(K_inv.dot(np.transpose(K_new[:,0])))

    Y_VAR_predict.append(y_var_predict)

plt.plot(X_new,Y_predict,'--',label='fonction de régression sur tout le nuage de point')

plt.legend()

plt.savefig('gaussian_processes_1d_fig_02.png', bbox_inches='tight')

                #Tracer la variance
plt.fill_between(X_new, [i-0.8*np.sqrt(Y_VAR_predict[idx]) for idx,i in enumerate(Y_predict)], 
[i+0.8*np.sqrt(Y_VAR_predict[idx]) for idx,i in enumerate(Y_predict)],color='#D3D3D3')

plt.savefig('gaussian_processes_1d_fig_03.png', bbox_inches='tight')