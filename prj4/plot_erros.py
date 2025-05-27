import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

M = 16 # ordem da constelacao

SNRt = np.linspace(-4, 12, 1000)
SNRlin = 10**(SNRt/10)

Pe = (3/(2*np.log2(M)))*(erfc(np.sqrt(np.log2(M)*SNRlin/10)))

# simular 0, 4, 8, 12 
snrs = [-3, 1, 5, 9]
pr = [0.212511, 0.120573, 0.0437698, 0.00488328]
pi = [0.401493, 0.359968, 0.3248677, 0.30292940]

plt.rcParams.update({'font.size':15})

plt.semilogy(SNRt, Pe, label = 'curva teórica')
plt.semilogy(snrs, pr, 'o:', label = 'canal plano')
plt.semilogy(snrs, pi, 'o:', label = 'canal com distorção')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid()
plt.legend()
plt.show()
