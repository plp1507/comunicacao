import numpy as np
import matplotlib.pyplot as plt

canal_plano = np.zeros(3000)
canal_plano[0] = 1

f_canal_plano = np.fft.fft(canal_plano)

canal_imag = np.asarray([0.19+.56, .45-1.28j, -.14-.53j, -.19+.23j, .33+.51j])
canal_imag = np.concatenate((canal_imag, np.zeros(3000 - len(canal_imag))))

f_canal_imag  = np.fft.fft(canal_imag)

plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0, 1, 3000), 10*np.log10(np.abs(f_canal_plano)))
ax[1].plot(np.linspace(0, 1, 3000), np.angle(f_canal_plano))
ax[0].grid()
ax[0].set_ylabel('Magnitude (dB)')
ax[1].grid()
ax[1].set_ylabel('Ângulo (rad)')
ax[1].set_xlabel('$f_k$')
plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(np.linspace(0, 1, 3000), 10*np.log10(np.abs(f_canal_imag)))
ax[1].plot(np.linspace(0, 1, 3000), np.angle(f_canal_imag))
ax[0].grid()
ax[0].set_ylabel('Magnitude (dB)')
ax[1].grid()
ax[1].set_ylabel('Ângulo (rad)')
ax[1].set_xlabel('$f_k$')
plt.show()
