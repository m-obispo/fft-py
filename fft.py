import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

#Aula 6a

#dados = np.genfromtxt('oscilador_caotico.csv')
#delT = dados[1,0]-dados[0,0]
#transfFourier = np.fft.fft(dados[:,2])
#numOnda = np.fft.fftfreq(len(dados), delT)
#espectro = np.abs(transfFourier)**2
#
#plt.subplot(2,1,1)
#plt.plot(dados[:,0],dados[:,2])
#plt.xlabel('t')
#plt.subplot(2,1,2)
#plt.plot(numOnda, espectro)
#plt.xlabel('k')
#plt.show()

#Aula 6b

def filtro(f, fc) :
    return 1/(1+abs(f/fc)**3)

def compressao(f,fc) :
    return (f<fc).astype(float)

img = image.imread('587705.jpg')
img = img.mean(axis=2)

#L=500
#img=np.array([[((i-L/2)**2+(j-L/2)**2<(L/3)**2)*255
#                   for i in range(L)] for j in range(L)])

ft = np.fft.fft2(img)
s = img.shape
d = 1
numOnda0 = np.repeat(np.fft.fftfreq(s[0],d),s[1]).reshape(s)
numOnda1 = np.tile  (np.fft.fftfreq(s[1],d),s[0]).reshape(s)
modk = np.sqrt(numOnda0**2+numOnda1**2)

#fc = [1/1000,1/100,1/30,1/10,0.3,1]
fc = [0.1,0.3,1]
for i in range(len(fc)) :
    f = filtro(modk, fc[i])
    #f = compressao(modk, fc[i])
    for passaBaixa in [True,False] :
        plt.subplot(2, len(fc), i+1+passaBaixa*len(fc))
        plt.title('%.3f - %.3f'%(fc[i], (f!=0).sum()/ft.size))
        plt.imshow(abs(np.fft.ifft2(ft*(f if passaBaixa else 1-f))),'gray')

plt.show()
