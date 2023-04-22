# 作者：songhy
# 作者微信：dabenmao3
# bilibili：同济七版抄7遍

import numpy as np
import matplotlib.pyplot as plt
from vmd import VMD
from mat_preprocessing import new_data



path=r'data'
new_data=new_data(path)
f=new_data['12k_Drive_End_B007_0_118.mat']
f=f[300:600]


T = 300
fs = 1/T
t = np.arange(1,T+1)/T
freqs = 2*np.pi*(t-0.5-fs)/(fs)
f_hat = np.fft.fftshift((np.fft.fft(f)))





alpha = 2000        # moderate bandwidth constraint
tau = 0             # noise-tolerance (no strict fidelity enforcement)
K = 4              # 3 modes
DC = 0              # no DC part imposed
init = 1            # initialize omegas uniformly
tol = 1e-6

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

plt.figure()
plt.plot(u.T)
plt.title('Decomposed modes')

sortIndex = np.argsort(omega[-1,:])
omega = omega[:,sortIndex]
u_hat = u_hat[:,sortIndex]
u = u[sortIndex,:]
linestyles = ['b', 'g', 'm', 'c', 'c', 'r', 'k']


fig1 = plt.figure()
plt.plot(t,f)
fig1.suptitle('Original input  ')

#  分解的中心频率 纵轴对应的迭代次数
fig2 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]))
plt.xlim(np.array([1,T/2])*np.pi*2)
ax = plt.gca()
ax.grid(which='major', axis='both', linestyle='--')
fig2.suptitle('Input signal spectrum')


fig3 = plt.figure()
for k in range(K):
    plt.semilogx(2*np.pi/fs*omega[:,k], np.arange(1,omega.shape[0]+1), linestyles[k])
fig3.suptitle('Evolution of center frequencies omega')


fig4 = plt.figure()
plt.loglog(freqs[T//2:], abs(f_hat[T//2:]), 'k:')
plt.xlim(np.array([1, T//2])*np.pi*2)
for k in range(K):
    plt.loglog(freqs[T//2:], abs(u_hat[T//2:,k]), linestyles[k])
fig4.suptitle('Spectral decomposition')
plt.legend(['Original','1st component','2nd component','3rd component'])

fig4 = plt.figure()

for k in range(K):
    plt.subplot(4,1,k+1)
    plt.plot(t,u[k,:], linestyles[k])

    plt.xlim((0,1))
    plt.title('Reconstructed mode %d'%(k+1))


fig5 = plt.figure()
fig5 = plt.figure().add_subplot(111)
fig5.plot(t, f, 'b', label='Original data')
fig5.plot(t, u[0]+u[1]+u[2]+u[3], 'r-', label='Processed data')


plt.show()



