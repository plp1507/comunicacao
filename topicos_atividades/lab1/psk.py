import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import erfc

### Funções
def decide(msg, constelacao):
    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constelacao.real, constelacao.imag))
    distcs = cdist(XA, XB, metric = 'euclidean')
    return constelacao[np.argmin(distcs, axis = 1)]

def psk_const(M):
    out = np.exp(1j*(np.pi/M)*np.arange(1, 2*M+1, 2))
    if(M == 2):
        out = np.sign(out.real)
    return out

### Constelações utilizadas
M = [2, 4, 8, 16]

### Características da SNR
npt = 6

SNRt = np.linspace(0, 20, 1000)
SNRs = np.linspace(0, 20, npt)

sigma2 = 10**(-SNRs/10)

SERt = np.zeros([len(M), 1000])
# curva teórica do BPSK e QPSK
SERt[0] = (1/2)*(erfc(np.sqrt(10**(SNRt/10))))
SERt[1] = erfc(np.sqrt(10**(SNRt/10))*np.sqrt(1/2))

# curva teórica do M-psk pra M > 4
for i in range(2, len(M)):
    SERt[i] = erfc(np.sqrt(10**(SNRt/10))*np.sin(np.pi/M[i]))

SERs = np.zeros([len(M), npt])

### Geração da mensagem
n = 10**6    # número de símbolos da simulação (por constelação)
mensagem = np.zeros([len(M)-1, n], dtype = 'complex')

# escolha aleatória de símbolos
mensagem_b = 2*np.round(np.random.rand(n)) - 1
for i in range(len(M)-1):
    mensagem[i] = np.random.choice(psk_const(M[i+1]), n)

### Simulação de Monte Carlo
mensagem_r = np.zeros([len(M)-1, n], dtype = 'complex')
mensagem_r_b = np.zeros(n)
sinal_rx = np.zeros([len(M)-1, n], dtype = 'complex')
sinal_rx_b = np.zeros(n)

for k, noise in enumerate(sigma2):
    fig, ax = plt.subplots(2, 2)
    w = np.sqrt(noise/2)*(np.random.randn(len(M)-1, n) + 1j*np.random.randn(len(M)-1, n))
    
    sinal_rx_b = mensagem_b + np.sqrt(noise/2)*(np.random.randn(n))  # ruido do bpsk
    sinal_rx = mensagem + w      # ruido das constelaçoes mpsk (complexas)

    # detecçao do bpsk - um pouco diferente do resto
    ax[0, 0].scatter(sinal_rx_b[:10**4], np.zeros(10**4))
    mensagem_r_b = np.sign(sinal_rx_b)
    SERs[0][k] = np.sum(mensagem_r_b != mensagem_b)/n
    ax[0, 0].scatter(mensagem_b[:100], np.zeros(100))
    ax[0, 0].grid()
    ax[0, 0].set_box_aspect(1)

    # detecçao dos M-psk
    for i in range(len(M)-1):
        indx = format(i+1, '02b')

        mensagem_r[i] = decide(sinal_rx[i], psk_const(M[i+1]))
        SERs[i+1][k] = np.sum(mensagem_r[i] != mensagem[i])/n
        
        ax[int(indx[0]), int(indx[1])].scatter(sinal_rx[i][:10**4].real, sinal_rx[i][:10**4].imag)
        ax[int(indx[0]), int(indx[1])].scatter(mensagem[i][:100].real, mensagem[i][:100].imag)
        ax[int(indx[0]), int(indx[1])].grid()
        ax[int(indx[0]), int(indx[1])].set_box_aspect(1)

    plt.show()


#Plot das curvas de SER

for i in range(len(M)):
    plt.semilogy(SNRt, SERt[i], label = f'{M[i]}PSK t')
    plt.semilogy(SNRs, SERs[i], 'o:', label = f'{M[i]}PSK s')

plt.legend()
plt.ylim(10**-7)
plt.ylabel('SER')
plt.xlabel('SNR (dB)')
plt.grid()
plt.show()
