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

def qam_const(M):
    range_ = int(np.sqrt(M))
    out = np.zeros([range_, range_], dtype = 'complex')
    for i in range(range_):
        for j in range(range_):
            out[i][j] = i + 1j*j
    out = np.reshape(out, M)
    out -= (np.sqrt(M)-1)*(.5 + .5j)
    eT = np.sum(np.abs(out)**2)/M
    out /= np.sqrt(eT)
    return out

### Constelações utilizadas
M = [4, 16, 64]
const4 = qam_const(M[0])
const16 = qam_const(M[1])
const64 = qam_const(M[2])


### Características da SNR
npt = 6

SNRt = np.linspace(0, 20, 1000)
SNRs = np.linspace(0, 20, npt)

sigma2 = 10**(-SNRs/10)

SERt = np.zeros([len(M), 1000])
for i in range(len(M)):
    argerfc = (3/(2*(M[i]-1)))*(10**(SNRt/10))
    p = (1-np.sqrt(1/M[i]))*erfc(np.sqrt(argerfc))
    SERt[i] = 1 - ((1-p)**2)

SERs = np.zeros([len(M), npt])

### Geração da mensagem
n = 10**6    # número de símbolos da simulação (por constelação)
mensagem = np.zeros([len(M), n], dtype = 'complex')

# escolha aleatória de símbolos
for k, i in enumerate(M):
    mensagem[k] = np.random.choice(qam_const(i), n)

### Simulação de Monte Carlo
mensagem_r = np.zeros([len(M), n], dtype = 'complex')
for k, noise in enumerate(sigma2):
    fig, ax = plt.subplots(1, len(M))
    w = np.sqrt(noise/2)*(np.random.randn(len(M), n) + 1j*np.random.randn(len(M), n))

    sinal_rx = mensagem + w

    for i in range(len(M)):
        ax[i].scatter(sinal_rx[i][:10**4].real, sinal_rx[i][:10**4].imag)
        if(i==0):
            mensagem_r[i] = decide(sinal_rx[i], const4)
        elif(i==1):
            mensagem_r[i] = decide(sinal_rx[i], const16)
        else:
            mensagem_r[i] = decide(sinal_rx[i], const64)

        SERs[i][k] = np.sum(mensagem_r[i] != mensagem[i])/n
        ax[i].scatter(mensagem[i].real, mensagem[i].imag)
        ax[i].grid()
        ax[i].set_box_aspect(1)
    plt.show()


#Plot das curvas de SER

for i in range(len(M)):
    plt.semilogy(SNRt, SERt[i], label = f'{M[i]}QAM t')
    plt.semilogy(SNRs, SERs[i], 'o:', label = f'{M[i]}QAM s')

plt.legend()
plt.ylim(10**-6)
plt.ylabel('SER')
plt.xlabel('SNR (dB)')
plt.grid()
plt.show()
