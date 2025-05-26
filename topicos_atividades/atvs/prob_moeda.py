import numpy as np
import matplotlib.pyplot as plt
from math import comb

def c(n, k):
    a = np.sum(np.log(np.arange(1, n+1)))
    b = np.sum(np.log(np.arange(1, k+1)))
    c = np.sum(np.log(np.arange(1, n-k+1)))
    return round(np.exp(a-b-c))

### Variáveis da simulação
M = 2*10**4      # número de simulações
n = 5*10**2      # número de realizações por simulação
prob = 0.2       # probabilidade de dar cara

### Simulação de Monte Carlo
# convenciona-se que cara é 1
realizacoes = np.random.choice([0, 1], [M, n], p = [1-prob, prob])

# soma-se a quantidade de caras e calcula o estimador de p
n_caras = np.sum(realizacoes, axis = 1)
n_caras = np.transpose(n_caras, -1)
ph = n_caras/n

### Distribuição teórica
k = np.arange(0, n+1)
p_t = np.zeros(len(k))
for i in range(len(k)):
    p_t[i] = M*c(n, k[i])*(prob**k[i])*(1-prob)**(n-k[i])

### Plot dos resultados
plt.hist(n_caras, bins = k[70: 131], label = 'Simulação de Monte Carlo')
plt.plot(k[70:131], p_t[70:131], '--', label = 'Distr. de prob. teórica')
plt.grid()
plt.legend()
plt.ylabel('Número de ocorrências')
plt.xlabel('Número de caras')

plt.show()
