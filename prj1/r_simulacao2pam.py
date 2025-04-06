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

x = np.pi*np.linspace(-l, l, 2*l*k + 1)/Ts

filtro = (1/Ts)*(np.sin(x*(1-alpha)) + 4*alpha*(x/np.pi)*np.cos(x*(1+alpha)))
filtro[x == 0] = (1/Ts)*(1+alpha*(4/np.pi - 1))
filtro[x != 0] /= x[x != 0]*(1-(4*alpha*(x[x != 0]/np.pi))**2)

filtro /= np.sqrt(np.sum(filtro**2))

#Processamento do sinal de entrada
imopen = plt.imread('lenna512.tif')
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

plt.plot(np.arange(0, 10*k, k), imagem_pam[:10])
plt.plot(x_h[l*k :l*k + 10*k])
plt.show()

#Passagem pelo canal
fc = (1/Ts)*(1+alpha)/2
b, a = signal.butter(10, 2*fc/Fs)

sinal_rx = signal.filtfilt(b, a, x_h)

#Adição de ruído
snr = 5    #(SNR em Eb/N0 dB)
pot_sinal_tx = (np.sum(sinal_rx)**2)/len(sinal_rx)
pot_sinal_tx = 10*np.log10(pot_sinal_tx)

pot_ruido = pot_sinal_tx-snr
print(pot_ruido)
pot_ruido = 10**(-pot_ruido/10)
print(pot_ruido)

w = pot_ruido*np.random.randn(len(sinal_rx))

sinal_rx += w
'''
plt.plot(sinal_rx[l*k//2 - 10: 10*k + l*k//2], label = 'sinal na entrada do receptor')
plt.plot(x_h[l*k//2 - 10: 10*k + l*k//2], ':', label = 'sinal na saída do transmissor')
plt.grid()
plt.legend()
plt.xlabel('n')
plt.show()

#Recepção
'''

x_casado = np.convolve(sinal_rx, np.flip(filtro))
plt.plot(x_casado[2*l*k - 1:10*k + 2*l*k])
plt.plot(x_h[l*k: l*k + 10*k])
plt.grid()
plt.show()


x_dwnsp = x_casado[2*l*k - 1: 2*l*k - 1 + len(imagem_pam)*k:k]
plt.plot(x_dwnsp)
plt.grid()
plt.show()
'''
plt.plot(imagem_pam[:10])
plt.plot(x_dwnsp[:10])
plt.grid()
plt.show()
'''
x_detec = []
for i in range(len(imagem_pam)):
    x_detec.append(int((np.sign(x_dwnsp[i]) + 1)/2))

'''
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

plt.stem(x_detec[:10])
plt.plot(imagem_bits[:10], ':')
plt.grid()
plt.show()
'''

BERs = np.sum(x_detec != imagem_bits)/len(imagem_pam)
BERt = (1/2)*erfc(np.sqrt(10**(snr/10)))

print(f"BER teórica: {BERt}")
print(f"BER simulada: {BERs}")

