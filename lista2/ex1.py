import numpy as np
import matplotlib.pyplot as plt

### medidas
dists_m = np.asarray([10, 20, 50, 100, 300])
pr_m = np.asarray([-70, -75, -90, -110, -125])

### estimativa
dists_t = np.linspace(1, 300, 1000)
pr_t = -31.5266 - 37.089296*np.log10(dists_t)# + 10*np.log10((10**(13.277482/10))*np.random.randn(1000))

plt.plot(dists_t, pr_t, label = 'Curva teórica')
plt.plot(dists_m, pr_m, 'o:', label = 'Medidas')
plt.grid()
plt.ylabel('Potência recebida (dBm)')
plt.xlabel('Distância (m)')
plt.legend()
plt.show()
