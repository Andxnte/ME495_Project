# cgle_numerical.py

import numpy as np
import matplotlib.pyplot as plt

# 1. 域与网格
L = 5.0
N = 256  # 空间方向网格数
x = np.linspace(-L, L, N, endpoint=False)
y = np.linspace(-L, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# 2. 时间步长与总时间
T = 1.0
dt = 0.01
n_steps = int(T / dt)

# 3. 物理参数
alpha = 0.5
beta = -0.5

# 4. 傅里叶算子
kx = (np.fft.fftfreq(N, d=(x[1]-x[0])) * 2 * np.pi).reshape(N, 1)
ky = (np.fft.fftfreq(N, d=(y[1]-y[0])) * 2 * np.pi).reshape(1, N)
lap = -(kx**2 + ky**2)

# 5. 初始条件：高斯包络
U = np.exp(-(X**2 + Y**2)).astype(np.complex128)

# 6. 构造 ETDRK4 系数
L_op = 1 + 1j*alpha * lap
E = np.exp(dt * L_op)
E2 = np.exp(dt * L_op / 2)

M = 16
r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
L_flat = (dt * L_op).reshape(N*N, 1)
LR = L_flat + r.reshape(1, M)
Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=1).reshape(N, N)
f1 = dt * np.mean((-4 - LR + np.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1).reshape(N, N)
f2 = dt * np.mean((2 + LR + np.exp(LR)*(-2 + LR)) / LR**3, axis=1).reshape(N, N)
f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR)*(4 - LR)) / LR**3, axis=1).reshape(N, N)

# 7. 时间推进
for _ in range(n_steps):
    N0 = -(1 + 1j*beta) * np.abs(U)**2 * U
    a = E2 * U + Q * N0
    Na = -(1 + 1j*beta) * np.abs(a)**2 * a
    b = E2 * U + Q * Na
    Nb = -(1 + 1j*beta) * np.abs(b)**2 * b
    c = E2 * a + Q * (2*Nb - N0)
    Nc = -(1 + 1j*beta) * np.abs(c)**2 * c
    U = E * U + f1 * N0 + 2*f2 * (Na + Nb) + f3 * Nc

# 8. 可视化实部
plt.figure(figsize=(6, 5))
plt.pcolormesh(x, y, np.real(U), shading='auto', cmap='viridis')
plt.colorbar(label='Re(A) at t=T')
plt.title('Numerical Solution (Real Part) at T=1.0')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("numerical_u_t1.png", dpi=150)
plt.show()

# Define the spatial domain for initial condition
L = 5.0
nx = ny = 200
x = np.linspace(-L, L, nx)
y = np.linspace(-L, L, ny)
X, Y = np.meshgrid(x, y)

# Initial condition: u(x, y, 0) = exp(-(x^2 + y^2)), v = 0
U0 = np.exp(-(X**2 + Y**2))
V0 = np.zeros_like(U0)

# Plot the real part of the initial condition (u)
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, U0, shading='auto', cmap='viridis')
plt.colorbar(label='u(x,y,0)')
plt.title('Initial Condition: u(x,y,0)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# Plot the imaginary part of the initial condition (v)
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, V0, shading='auto', cmap='viridis')
plt.colorbar(label='v(x,y,0)')
plt.title('Initial Condition: v(x,y,0)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()