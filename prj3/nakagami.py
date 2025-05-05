import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, hyp2f1, gamma

#### Parâmetro m e distribuição Nakagami-m
m = [0.5, 2, 15, 1]

#### Configurações da SNR
npt = 10
SNRt = np.linspace(-2, 10, 1000, endpoint = True)
SNRs = SNRt[::1000//(npt-1)]
sigma2 = 10**(-SNRs/10)

#### Geração da curva teórica
BERt = np.zeros([4, 1000])
for i in range(4):
    BERt[i] = gamma(m[i] + 1/2)/(2*np.sqrt(np.pi)*gamma(m[i] + 1))
    BERt[i] *= (1 + (10**(SNRt/10))/m[i])**(-m[i])
    BERt[i] *= hyp2f1(m[i], 1/2, m[i] + 1, 1/(1 + 10**((SNRt/10))/m[i]))

BERs = np.zeros([4, npt])

#### Simulação de Monte Carlo
n = 10**5     # número de símbolos transmitidos
mensagem = np.sign(np.random.rand(n) - 0.5)   # geração dos símbolos BPSK
sig2 = 1      # variância do canal

BERs_awgn = np.zeros(npt)

for i in range(4):
    rand_n = 4*np.random.rand(n)
    canal = (2*m[i]**m[i])*rand_n**(2*m[i] - 1)
    canal /= gamma(m[i])*sig2**m[i]
    canal *= np.exp((-m[i]*rand_n**2)/sig2)

    for k, noise in enumerate(sigma2):
        w = np.sqrt(noise/2)*np.random.randn(n)
        sinal_rx = canal*mensagem + w
        
        sinal_rx_awgn = mensagem + w

        mensagem_r = np.sign(sinal_rx)
        BERs[i][k] = np.sum(mensagem_r != mensagem)/n
        BERs_awgn[k] = np.sum(np.sign(sinal_rx_awgn) != mensagem)/n


plt.semilogy(SNRt, (1/2)*erfc(np.sqrt(10**(SNRt/10))))
plt.semilogy(SNRs, BERs_awgn, 'o:', label = 'AWGN')
'''
for i in range(3):
    plt.semilogy(SNRt, BERt[i], label = f'Nakagami-m: m = {m[i]}')
    plt.semilogy(SNRs, BERs[i], 'o:', label = f'N-m simulado: m = {m[i]}')

plt.semilogy(SNRt, BERt[3], label = 'Rayleigh')
plt.semilogy(SNRs, BERs[3], 'o:', label = 'Rayleigh simulado')
'''
plt.legend()
plt.grid()
plt.show()

for l in range(4):
    nakagami = (2*m[l]**m[l])*(np.linspace(0, 2, 1000)**(2*m[l] - 1))*np.exp((-m[l]/sig2)*np.linspace(0, 2, 1000)**2)/(gamma(m[l])*sig2**m[l])
    plt.plot(np.linspace(0, 2, 1000), nakagami, label = f'm = {m[l]}')

plt.legend()
plt.grid()
plt.show()
