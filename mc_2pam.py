import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy.special import erfc
from scipy.spatial.distance import cdist

def decide(msg, constell):
    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constell.real, constell.imag))
    distcs = cdist(XA, XB, metric = 'euclidean')

    return constell[np.argmin(distcs, axis = 1)]

## Configurações do filtro de transmissão

alpha = 1
k = 10
Fs = 100
Ts = k/Fs
l = 10

x = np.linspace(-l/2, l/2, l*Fs + 1)/Ts
filtro_rrc = (1/Ts)*(np.sin(np.pi*x*(1-alpha)) + (4*alpha*x)*np.cos(np.pi*x*(1+alpha)))
filtro_rrc[x==0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro_rrc[x!=0] /= np.pi*x[x!=0]*(1-(4*alpha*x[x!=0])**2)
filtro_rrc[abs(x)==0.25*Ts/alpha] = (alpha/(Ts*np.sqrt(2)))*((1+2/np.pi)*np.sin(0.25*np.pi/alpha)+(1-2/np.pi)*(np.cos(0.25*np.pi/alpha)))

filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

## Geração e modulação da mensagem transmitida
n = 5*10**5 # Número de símbolos a serem transmitidos

constelacao_pam2 = np.linspace(-2 + 1, 2 - 1, 2) # Pontos da constelação 2-PAM
constelacao_pam4 = np.linspace(-4 + 1, 4 - 1, 4) # Pontos da constelação 4-PAM
constelacao_pam8 = np.linspace(-8 + 1, 8 - 1, 8) # Pontos da constelação 8-PAM

eT = np.sum(np.abs(constelacao_pam2)**2)/2        # Normalização pela energia média da constelação
constelacao_pam2 /= np.sqrt(eT)
eT = np.sum(np.abs(constelacao_pam4)**2)/4        # Normalização pela energia média da constelação
constelacao_pam4 /= np.sqrt(eT)
eT = np.sum(np.abs(constelacao_pam8)**2)/8        # Normalização pela energia média da constelação
constelacao_pam8 /= np.sqrt(eT)

## Transmissão
# Escolha aleatória de n símbolos da constelação M-PAM
mensagem_original = np.zeros([3, n])
mensagem_original[0] = np.random.choice(constelacao_pam2, n)
mensagem_original[1] = np.random.choice(constelacao_pam4, n)
mensagem_original[2] = np.random.choice(constelacao_pam8, n)

m_ups = np.zeros([3, n*k])
m_conv = np.zeros([3, n*k + l*Fs])

for i in range(3):
    m_ups[i][::k] = mensagem_original[i]
    m_conv[i] = np.convolve(m_ups[i], filtro_rrc)

## Passagem pelo canal de comunicação (filtro passa-baixa)
fc = (0.5/Ts)*(1+alpha)                 # Banda estimada
canalB, canalA = butter(10, 2*fc/Fs)  # Taps do filtro do canal

sinal_rx = np.zeros([3, n*k + l*Fs])
for i in range(3):
    sinal_rx[i] = lfilter(canalB, canalA, m_conv[i])

## Simulação de Monte Carlo da recepção
# Configurações da SNR

npt_sim = 5
SNRt = np.zeros([3, 1000])
SNRt[0] = np.linspace(0, 10, 1000) # Ranges da SNR da simulação
SNRt[1] = np.linspace(3, 13, 1000) # Ranges da SNR da simulação
SNRt[2] = np.linspace(6, 16, 1000) # Ranges da SNR da simulação
SERt = np.zeros([3, 1000])

for i in range(1, 4):
    M = 2**i
    SERt[i-1] = (1 - 1/M)*erfc(np.sqrt(10**(SNRt[i-1]/10))*np.sqrt(3/(M**2 - 1))) # Curva teórica da SER

SNRs = np.zeros([3, npt_sim + 1])
sigma2 = np.zeros([3, npt_sim + 1])

for i in range(3):
    SNRs[i] = SNRt[i][::1000//npt_sim - 1]
    sigma2[i] = 10**(-SNRs[i]/10)

SERs = np.zeros([3, len(sigma2[0])])

y = np.zeros([3, n*k + l*Fs])
m_casado = np.zeros([3, n*k + 2*l*Fs])
m_dwnsp = np.zeros([3, n])
m_reconstruida = np.zeros([3, n])

for g in range(3):
    for j, noise in enumerate(sigma2[g]):
        w = np.sqrt(noise/2)*np.random.randn(n*k + l*Fs)
    
        y[g] = sinal_rx[g] + w
        m_casado[g] = np.convolve(y[g], filtro_rrc)
        m_dwnsp[g] = m_casado[g][l*Fs+10:l*Fs + n*k+10:k]

        if(g==0):
            m_reconstruida[g] = decide(m_dwnsp[g], constelacao_pam2)
        if(g==1):
            m_reconstruida[g] = decide(m_dwnsp[g], constelacao_pam4)
        if(g==2):
            m_reconstruida[g] = decide(m_dwnsp[g], constelacao_pam8)

        SERs[g][j] = np.sum(m_reconstruida[g] != mensagem_original[g])/n

for i in range(3):
    plt.semilogy(SNRt[i], SERt[i], label = f'{2**(i+1)} - PAM (teórico)')
    plt.semilogy(SNRs[i], SERs[i], 'o:', label = f'{2**(i+1)} - PAM')

plt.legend()
plt.grid()
plt.show()
