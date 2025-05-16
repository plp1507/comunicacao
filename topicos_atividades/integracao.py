import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

### Funções
def invQ(x):
    return np.sqrt(2)*erfcinv(2*x)

### Variáveis da simulação
n = 10**4      # número de realizações por segmento

### Simulação de Monte Carlo

realizacoes = 1*np.random.rand(n)
result = 0.7468

I = np.zeros(n)
for i in range(n):
    I[i] = np.sum(np.exp(-realizacoes[:i+1]**2))/(i+1)


# Intervalo de confiança
conf = 0.2

### Configurações do plot

plt.plot(I, label = 'simulação de Monte Carlo')
plt.plot(result*np.ones(n), '--', label = 'resultado teórico')
plt.plot(result + invQ(conf/2)*0.201/np.sqrt(np.arange(1, n+1)), 'g:')
plt.plot(result - invQ(conf/2)*0.201/np.sqrt(np.arange(1, n+1)), 'g:')

plt.ylabel('resultado da integral')
plt.xlabel('número de segmentos')

plt.grid()
plt.legend()

plt.show()
