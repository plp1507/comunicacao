import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.signal import upfirdn

def rrc(alpha, Tb, k, l):
    
    # Definicao do filtro RRC:
    #   alpha > fator de roll off
    #   Tb    > periodo do simbolo
    #   k     > fator de oversampling, comprimento do bloco de simbolo
    #   l     > duracao do simbolo (em blocos de simbolo)

    Ts = Tb/k

    t = np.arange(-l*Tb/2, l*Tb/2 + Ts, Ts)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        filtro_rrc = np.divide(np.sin(np.pi*t*(1-alpha)/Tb) +
                (4*alpha*t/Tb)*np.cos(np.pi*t*(1+alpha)/Tb),
                        (np.pi*t/Tb)*(1-(4*alpha*t/Tb)**2))

    filtro_rrc /= Tb

    filtro_rrc[np.argwhere(np.isnan(filtro_rrc))] = (1/Tb)*(1 - alpha + 4*alpha/np.pi)
    filtro_rrc[np.argwhere(np.isinf(filtro_rrc))] = ((alpha/(Tb*np.sqrt(2))) * ((1 + (2/np.pi))*np.sin(np.pi/(4*alpha)) + (1 - (2/np.pi))*np.cos(np.pi/(4*alpha))))

    atraso = l*Tb//2

    filtro_rrc /= np.sqrt(np.sum(filtro_rrc**2))

    return atraso, filtro_rrc

def decide(msg, constelacao):
    # Funcao de decisao
    #   msg > simbolos recebidos
    #   constelacao > constelacao utilizada

    XA = np.column_stack((msg.real, msg.imag))
    XB = np.column_stack((constelacao.real, constelacao.imag))

    distcs = cdist(XA, XB, metric = 'euclidean')
    return constelacao[np.argmin(distcs, axis = 1)]

## geracao do filtro de transmissao
T = 1             #período de amostragem
l_bloco = 21      #comprimento do bloco de simbolo
Ts = l_bloco*T    #período do bloco de símbolo

# filtro rrc com alpha 0.5 durando 20 simbolos
g_delay, f_rrc = rrc(0.5, Ts, l_bloco, 20)

## conversao da imagem para serie de bits
imagem_o = plt.imread('./lenna512.tif')
imagem_o_serie = np.reshape(imagem_o, -1)
imagem_o_bits = np.zeros(8*len(imagem_o_serie))

for i in range(len(imagem_o_serie)):
    # conversao do valor de cada pixel em numeros binarios
    bits_str = format(imagem_o_serie[i], '08b')
    imagem_o_bits[8*i:8*(i+1)] = np.asarray(list(bits_str))

## mapeamento dos bits para serie de simbolos
simbolo_bb = np.zeros(len(imagem_o_bits)//4, dtype = 'complex')

# implementação do mapeamento gray
for i in range(len(simbolo_bb)):
    conj_bits = imagem_o_bits[4*i:4*(i+1)]
    
    sigr = np.sign(conj_bits[0] - 0.5)
    sigi = np.sign(conj_bits[2] - 0.5)

    distr = 3/2 - conj_bits[1]
    disti = 3/2 - conj_bits[3]

    # construcao dos simbolos em banda base
    simbolo_bb[i] = sigr*distr + 1j*sigi*disti

constelacao = np.unique(simbolo_bb)

## passagem pelo filtro de transmissao
simbolo_rrc = upfirdn(f_rrc, simbolo_bb, l_bloco)

# calculo da potencia do sinal transmitido
p_mensagem = np.sqrt(np.mean(np.abs(simbolo_rrc)**2))

## Passagem pelo canal de comunicacao
SNR = 12        # SNR em dB
pot_w = p_mensagem*(10**(-SNR/10))*l_bloco/2 # potencia de ruido

print(10*np.log10(p_mensagem/pot_w))

w = np.sqrt(pot_w/2)*(np.random.randn(len(simbolo_rrc)) + 1j*np.random.randn(len(simbolo_rrc)))

print(10*np.log10(p_mensagem/np.mean(abs(w)**2)))

# canal com distorcao
canal = np.asarray([0.19 + 0.56j, 0.45 - 1.28j, -0.14 - 0.53j, -0.19 + 0.23j, 0.33 + 0.51j])

sinal_rx = np.convolve(simbolo_rrc, canal)[:len(simbolo_rrc)] + w

## Recepcao
simbolo_recep = np.convolve(sinal_rx, f_rrc)
simbolo_dwnsp = simbolo_recep[2*g_delay:-2*g_delay:l_bloco]

simbolo_r = decide(simbolo_dwnsp, constelacao)
'''
# Scatter plot
plt.rcParams.update({'font.size': 15})

plt.scatter(simbolo_dwnsp.real, simbolo_dwnsp.imag, label = 'símbolos recebidos')
plt.scatter(simbolo_bb.real, simbolo_bb.imag, label = 'constelação original')
plt.grid()
plt.xlabel('I')
plt.ylabel('Q')
plt.legend()
plt.show()

# Diagrama de olho

fig, ax = plt.subplots(1, 2, figsize = (10, 5))
for i in range(300):
    ax[0].plot(simbolo_recep[2*g_delay + i*l_bloco + l_bloco//2 + 2:2*g_delay + (i+1)*l_bloco + l_bloco//2 - 1].real + 2, 'k')
    ax[1].plot(simbolo_recep[2*g_delay + i*l_bloco + l_bloco//2 + 2:2*g_delay + (i+1)*l_bloco + l_bloco//2 - 1].imag + 2, 'k')

ax[0].set_xlabel('parte real')
ax[1].set_xlabel('parte imaginária')
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[0].grid()
ax[1].grid()
plt.show()
'''
## Transformacao dos simbolos recebidos em bits 
bits_r = np.zeros(len(imagem_o_bits))
for i in range(len(simbolo_r)):
    real = simbolo_r[i].real
    imag = simbolo_r[i].imag
    
    bit0 = (np.sign(real) == 1)
    bit2 = (np.sign(imag) == 1)

    bit1 = (np.abs(real) < 1)
    bit3 = (np.abs(imag) < 1)

    bits_r[4*i:4*(i+1)] = np.asarray([bit0, bit1, bit2, bit3])

print(f'Erro de bits para SNR {SNR} dB: {np.sum(bits_r != imagem_o_bits)}')
print(f'Taxa de erro de bits: {np.sum(bits_r != imagem_o_bits)/len(imagem_o_bits)}')

# Recuperacao da imagem
pixels_r = np.zeros(len(imagem_o_serie))
for i in range(len(imagem_o_serie)):
    byte = ''
    for j in range(8):
        byte += (str(int(bits_r[8*i + j])))

    pixels_r[i] = int(byte, 2)

imagem_r = np.reshape(pixels_r, np.shape(imagem_o))

'''
### Plots
fig, ax = plt.subplots(2, 1)
ax[0].imshow(imagem_o)
ax[1].imshow(imagem_r)
ax[0].set_xlabel('imagem original')
ax[1].set_xlabel('imagem recebida')
plt.show()
'''

