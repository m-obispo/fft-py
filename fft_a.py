import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

#Aula 6a

dados = np.genfromtxt('oscilador_caotico.csv')
delT = dados[1,0]-dados[0,0]
transfFourier = np.fft.fft(dados[:,2])
numOnda = np.fft.fftfreq(len(dados), delT)
espectro = np.abs(transfFourier)**2

plt.subplot(2,1,1)
plt.plot(dados[:,0],dados[:,2])
plt.xlabel('t')
plt.subplot(2,1,2)
plt.plot(numOnda, espectro)
plt.xlabel('k')
plt.show()