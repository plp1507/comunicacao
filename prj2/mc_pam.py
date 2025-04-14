import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.special import erfc
from scipy.spatial.distance import cdist

def decide(msg, constel):
    ##Função de hard decide
    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constel.real, constel.imag))
    distcs = cdist(XA, XB, metric= 'euclidean')
    return constel[np.argmin(distcs, axis= 1)]

##Configurações do filtro de transmissão

alpha = 1
k = 10
Fs = 100
Ts = k/Fs
l = 10

x = np.linspace(-l/2, l/2, l*Fs + 1)/Ts

filtro_rrc = (1/Ts)*(np.sin(np.pi*x*(1-alpha)) + (4*alpha*x)*np.cos(np.pi*x*(1+alpha)))
filtro_rrc[x==0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro_rrc[x!=0] /= np.pi*x[x!=0]*(1-(4*alpha*x[x!=0])**2)

filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

##Configurações da SNR

npt = 7

SNRt = np.zeros([3, 1000])          #SNR teórica
SNRt[0] = np.linspace(0, 8, 1000)
SNRt[1] = np.linspace(0, 8, 1000)
SNRt[2] = np.linspace(0, 8, 1000)

SERt = np.zeros([3, 1000])

#Curvas teóricas da SER
for i in range(3):
    M = 2**(i+1)
    SERt[i] = (1 - 1/M)*erfc(np.sqrt(10**(SNRt[i]/10))*np.sqrt(3/(M**2 - 1)))

SNRs = np.zeros([3, npt + 1])
sigma2 = np.zeros([3, npt + 1])

for i in range(3):
    SNRs[i] = SNRt[i][::1000//npt -1]
    sigma2[i] = 10**(-SNRs[i]/10)

SERs = np.zeros([3, npt + 1])

##Configurações da mensagem transmitida
n = 10**5       #Número de símbolos transmitidos

constelacao_2pam = np.linspace(-2+1, 2-1, 2)
constelacao_4pam = np.linspace(-4+1, 4-1, 4)
constelacao_8pam = np.linspace(-8+1, 8-1, 8)
constelacao_4pam /= np.sqrt(np.sum(constelacao_4pam**2)/len(constelacao_4pam)) #normalização
constelacao_8pam /= np.sqrt(np.sum(constelacao_8pam**2)/len(constelacao_8pam)) #normalização

#Transmissão
mensagem_original = np.zeros([3, n])
mensagem_original[0] = np.random.choice(constelacao_2pam, n)
mensagem_original[1] = np.random.choice(constelacao_4pam, n)
mensagem_original[2] = np.random.choice(constelacao_8pam, n)

m_ups = np.zeros([3, n*k])
m_conv = np.zeros([3, n*k + l*Fs])
for i in range(3):
    m_ups[i][::k] = mensagem_original[i]
    m_conv[i] = np.convolve(m_ups[i], filtro_rrc)

sinal_rx = np.zeros([3, n*k + l*Fs])

fc = (0.5/Ts)*(1+alpha)
canalb, canala = butter(10, 2*fc/Fs)

for i in range(3):
    sinal_rx[i] = lfilter(canalb, canala, m_conv[i])

#Recepção
y = np.zeros([3, n*k + l*Fs])
m_casado = np.zeros([3, 2*l*Fs + n*k])
m_dwnsmp = np.zeros([3, n])
m_recons = np.zeros([3, n])

w = np.ones([3, n*k + l*Fs])

off = 10

for g in range(3):
    for j, noise in enumerate(sigma2[g]):
        w[g] = np.sqrt(noise/2)*np.random.randn(n*k + l*Fs)

        y[g] = sinal_rx[g] + w[g]
        m_casado[g] = np.convolve(y[g], filtro_rrc)
        m_dwnsmp[g] = m_casado[g][l*Fs+off:l*Fs+n*k+off:k]
        
        if(g==0):
            m_recons[g] = decide(m_dwnsmp[g], constelacao_2pam)
        if(g==1):
            m_recons[g] = decide(m_dwnsmp[g], constelacao_4pam)
        if(g==2):
            m_recons[g] = decide(m_dwnsmp[g], constelacao_8pam)

        SERs[g][j] = np.sum(m_recons[g] != mensagem_original[g])/n

for i in range(3):
    plt.semilogy(SNRt[i], SERt[i], label = f'{2**(i+1)}-PAM (teórico)')
    plt.semilogy(SNRs[i], SERs[i], 'o:', label = f'{2**(i+1)}-PAM')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')

plt.grid()
plt.legend()
plt.show()
