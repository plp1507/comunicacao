import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfcinv

### Funções
def invQ(x):
    # Função Q inversa
    return np.sqrt(2)*erfcinv(2*x)

### Variáveis da simulação
n = 1*10**2      # número de realizações por segmento
M = 3*10**4      # número de segmentos

### Simulação de Monte Carlo

# Realizações da distr. aleatória
realizacoes = 1*np.random.rand(M*n)
realizacoes = np.reshape(realizacoes, [M,n])

# Cálculo da integral pela coleta de amostras aleatórias
I_n = np.zeros(M)
for i in range(M):
    I_n[i] = np.sum(np.exp(-realizacoes[i]**2))/n


# Cálculo da média acumulativa
I = np.zeros(M)
for i in range(M):
    I[i] = np.sum(I_n[:i+1])/(i+1)

# Intervalo de confiança e resultado da integral
conf = 0.8
result = 0.7468

### Configurações do plot

plt.plot(I, label = 'simulação de Monte Carlo')
plt.plot(result*np.ones(M), '--', label = 'resultado teórico')
plt.plot(result + invQ(conf/2)*0.201/np.sqrt(np.arange(1, M+1)), 'g:')
plt.plot(result - invQ(conf/2)*0.201/np.sqrt(np.arange(1, M+1)), 'g:')

plt.ylim(result - 0.01, result + 0.01)

plt.ylabel('resultado da integral')
plt.xlabel('número de segmentos')
plt.title('Integração de Monte Carlo da expressão $e^{-x^2}$')

plt.grid()
plt.legend()

plt.show()
