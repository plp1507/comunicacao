import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


EbN0 = np.linspace(0, 12, 1000)
PSK8 = (1/3)*erfc(np.sqrt((10**(EbN0/10))*3)*np.sin(np.pi/8))
PSK4 = (1/2)*erfc(np.sqrt(10**(EbN0/10)))

plt.semilogy(EbN0, PSK8, label ='8-PSK')
plt.semilogy(EbN0, PSK4, label = 'QPSK')
plt.xlabel('EbN0 (dB)')
plt.ylabel('BER')
plt.ylim(5*10**-4)
plt.grid()
plt.legend()
plt.show()
