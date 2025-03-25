import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.special import erfc

##Parâmetros do filtro
k = 10
alpha = 0.5
Fs = 1000
Ts = k/Fs
l = 25

x = np.pi*np.linspace(-l*Ts/2, l*Ts/2, l*k)/Ts

filtro = (1/Ts)*(np.sin(x*(1-alpha)) + 4*alpha*(x/np.pi)*np.cos(x*(1+alpha)))
filtro[x == 0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro[x != 0] /= x[x != 0]*(1-(4*alpha*(x[x != 0]/np.pi))**2)

filtro /= np.sqrt(2*np.sum(filtro**2))

#Processamento do sinal de entrada
imopen = plt.imread('imagens/lenna512.tif')
imagem = np.reshape(imopen, -1)

imagem_bits = np.zeros(len(imagem)*8)

for i in range(len(imagem)):
    for j in range(8):
        aux = format(imagem[i], '08b')
        imagem_bits[8*i+j] = int(aux[j])

imagem_pam = 2*imagem_bits - 1

x_ups = np.zeros(len(imagem_pam)*k)
x_ups[::k] = imagem_pam

x_h = np.convolve(x_ups, filtro)

#Passagem pelo canal
fc = (1/Ts)*(1+alpha)/2
b, a = signal.butter(10, 2*fc/Fs)

imp = np.zeros(100); imp[0] = 1
b1, a1 = signal.butter(20, 2*fc/Fs)

filtro_fft = 10*np.log10(np.fft.fft(signal.lfilter(b1, a1, imp)))

sinal_rx = signal.filtfilt(b, a, x_h)

#Adição de ruído
snr = 9    #(SNR em Eb/N0 dB)
pot_sinal_tx = (np.sum(sinal_rx)**2)/len(sinal_rx)
pot_sinal_tx = 10*np.log10(pot_sinal_tx)

print(pot_sinal_tx)

pot_ruido = pot_sinal_tx-snr

w = np.sqrt(10**(pot_ruido/10))*np.random.randn(len(sinal_rx))

print(10*np.log10(np.mean(np.abs(w)**2)))

sinal_rx += w

plt.plot(sinal_rx[l*k//2 - 10: 10*k + l*k//2], label = 'sinal na entrada do receptor')
plt.plot(x_h[l*k//2 - 10: 10*k + l*k//2], ':', label = 'sinal na saída do transmissor')
plt.grid()
plt.legend()
plt.xlabel('n')
plt.show()




#Recepção

x_casado = np.convolve(sinal_rx, np.flip(filtro))
x_dwnsp = x_casado[l*k - 1: l*k - 1 + len(imagem_pam)*k:k]

x_detec = []
for i in range(len(imagem_pam)):
    x_detec.append(int((np.sign(x_dwnsp[i]) + 1)/2))

imagem_r = np.zeros(len(imagem))
for i in range(len(imagem)):
    byte = x_detec[8*i: 8*(i+1)]
    byte_str = ""
    for j in range(8):
        byte_str += str(byte[j])

    imagem_r[i] = int(byte_str, 2)

imagem_r = np.reshape(imagem_r, np.shape(imopen))

fig, ax = plt.subplots(1,2)
ax[0].imshow(imopen, cmap = 'gray')
ax[1].imshow(imagem_r, cmap = 'gray')
plt.show()


########diagrama de olho
sig = x_casado[l*k - 1:l*k-1+(len(imagem_pam))*k]

fig, ax = plt.subplots()
for i in range(200):
    #ax.plot(np.linspace(-Ts/2, Ts/2, k*l), sig[i*k:(i+2)*k], 'k')
    ax.plot(np.linspace(-Ts/2, Ts/2, 2*k + 1), sig[i*k:(i+2)*k+1], 'k')

ax.set_box_aspect(1)
ax.set_xlabel('t')
plt.grid()
plt.show()


BERs = np.sum(x_detec != imagem_pam)/len(x_detec)
BERt = (1/2)*erfc(np.sqrt(10**(snr/10)))

print(f"BER teórica: {BERt}")
print(f"BER simulada: {BERs}")

