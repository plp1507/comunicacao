import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

### Variáveis da simulação
n = 7*10**5     # número de símbolos

mensagem = 2*np.round(np.random.rand(n)) - 1   # geração de símbolos aleatórios BPSK

### Características da SNR
npt = 10

SNRt = np.linspace(-2, 10, 1000)
SNRs = SNRt[::1000//(npt-1)]

sigma2 = 10**(-SNRs/10)

BERt = (1/2)*erfc(np.sqrt(10**(SNRt/10)))
BERs = np.zeros(npt)

### Simulação de Monte Carlo
for k, noise in enumerate(sigma2):
    w = np.sqrt(noise/2)*np.random.randn(n)
    y = mensagem + w

    BERs[k] = np.sum(np.sign(y) != mensagem)/n

### Configurações do plot
plt.semilogy(SNRt, BERt, label = 'BER teórica')
plt.semilogy(SNRs, BERs, 'o:', label = 'BER simulada')

plt.ylabel('BER')
plt.xlabel('SNR (dB)')

plt.grid()
plt.legend()

plt.show()
