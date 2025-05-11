import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.special import hyp2f1, gamma

def nakagami_rand(m, n):
    range_n = np.linspace(0, 1, 10000)
    random_nakagami = 2*(m**m)*range_n**(2*m - 1)*np.exp(-m*range_n**2)/gamma(m)

    return np.random.choice(random_nakagami, n)

#### Características da simulação AWGN
## SNR
npt = 10

SNRt = np.linspace(-5, 10, 1000)
SNRs = SNRt[::1000//(npt-1)]

sigma2 = 10**(-SNRs/10)

SERt_AWGN = (1/2)*erfc(np.sqrt(10**(SNRt/10)))
SERs_AWGN = np.zeros(npt)

## Geração da mensagem
n = 5*10**5
mensagem = np.sign(np.random.rand(n) - .5)

##Simulação de Monte Carlo
for k, noise in enumerate(sigma2):
    w = np.sqrt(noise/2)*np.random.randn(n)

    sinal_rx_AWGN = mensagem + w
    sinal_demod_AWGN = np.sign(sinal_rx_AWGN)

    SERs_AWGN[k] = np.sum(sinal_demod_AWGN != mensagem)/n

plt.semilogy(SNRt, SERt_AWGN, label = 'AWGN')
'''
plt.semilogy(SNRs, SERs_AWGN, 'o:')
plt.grid()
plt.show()
'''

#### Características da simulação Nakagami-m
m = [1, 0.5, 2, 15]

## SER teórica
SERs_nm = np.zeros([4, npt])
SERt_nm = np.zeros([4, 1000])
for i in range(4):
    SERt_nm[i] = (gamma(m[i] + 0.5)/(2*np.sqrt(np.pi)*gamma(m[i] + 1)))
    SERt_nm[i] *= (1 + (10**(SNRt/10))/m[i])**-m[i]
    SERt_nm[i] *= hyp2f1(m[i], 0.5, m[i] + 1, 1/(1+ (10**(SNRt/10))/m[i]))

## Simulação de Monte Carlo
for o in range(4):
    canal = nakagami_rand(m[o], n)

    for k, noise in enumerate(sigma2):
        w = np.sqrt(noise/2)*np.random.randn(n)

        sinal_rx_nm = canal*mensagem + w
        
        sinal_demod_nm = np.sign(sinal_rx_nm/canal)

        SERs_nm[o][k] = np.sum(sinal_demod_nm != mensagem)/n

    plt.semilogy(SNRs, SERs_nm[o], 'd:', label = f'NM simulado: m = {m[o]}')
    plt.semilogy(SNRt, SERt_nm[o], label = f'NM teórico: m = {m[o]}')

plt.grid()
plt.legend()
plt.show()

