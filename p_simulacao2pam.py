import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.spatial.distance import cdist

def decide(msg, constel):
    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constel.real, constel.imag))
    distcs = cdist(XA, XB, metric ='euclidean')
    return constel[np.argmin(distcs, axis=1)]

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
imopen = plt.imread('./imagens/shuttle_80x60.tif')
imagem = np.reshape(imopen, -1)

imagem_bits = np.zeros(len(imagem)*8)

for i in range(len(imagem)):
    for j in range(8):
        aux = format(imagem[i], '08b')
        imagem_bits[8*i+j] = int(aux[j])

imagem_pam = np.zeros(len(imagem_bits)//2)

constelacao_4pam = np.asarray([-3, -1, 1, 3])

for i in range(len(imagem_bits)//2):
    simbolo = int(imagem_bits[2*i]*2 + imagem_bits[2*i+1])
    imagem_pam[i] = constelacao_4pam[simbolo]

x_ups = np.zeros(len(imagem_pam)*k)
x_ups[::k] = imagem_pam

fig, ax = plt.subplots(2, 1)
ax[0].stem(x_ups[:100])
ax[0].grid()
ax[0].set_xlabel('n')

x_h = np.convolve(x_ups, filtro)

ax[1].plot(x_h[l*k//2 - 10: 10*k + l*k//2])
ax[1].grid()
ax[1].set_xlabel('n')
plt.show()


plt.plot(np.linspace(-0.5, 0.5, len(x_h)), np.fft.fftshift(np.abs(np.fft.fft(x_h))))
plt.grid()
plt.xlabel('k')
plt.show()

#Passagem pelo canal
fc = (1/Ts)*(1+alpha)/2
b, a = signal.butter(10, 2*fc/Fs)

sinal_rx = signal.filtfilt(b, a, x_h)

#Recepção

x_casado = np.convolve(sinal_rx, np.flip(filtro))
x_dwnsp = x_casado[l*k -1: l*k-1 + len(imagem_pam)*k:k]
x_detec = np.zeros(len(imagem_pam))

im_bits_r = np.zeros(2*len(imagem_pam))
for i in range(len(imagem_pam)):
    x_detec[i] = int(decide(x_dwnsp[i], constelacao_4pam)[0])
    simb_norm = (x_detec[i] + 3)//2
    im_bits_r[2*i] = simb_norm//2
    im_bits_r[2*i + 1] = simb_norm % 2


print(np.sum(x_detec!=imagem_pam)/len(imagem_pam))
print(np.sum(im_bits_r!=imagem_bits)/len(im_bits_r))

imagem_r = np.zeros(len(imagem))

for i in range(0, len(imagem)):
    byte = im_bits_r[8*i: 8*(i+1)]
    byte_str = ""
    for j in range(8):
        byte_str += str(byte[j])[0]
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


