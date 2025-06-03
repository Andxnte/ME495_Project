#!/usr/bin/env python3
import deepxde as dde
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

# 禁用 TF2 行为，使用 TF1.x API
tf.disable_v2_behavior()

# 1. 定义空间-时间域：二维空间 [-L,L]^2 和时间区间 [0,T]
L = 5.0
T = 1.0
space = dde.geometry.Rectangle(xmin=[-L, -L], xmax=[L, L])
time_domain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(space, time_domain)

# 2. 声明可训练的物理参数 alpha, beta
alpha = dde.Variable(0.1)
beta = dde.Variable(0.1)

# 3. 定义 CGLE 的 PDE 残差算子
def CGLE_pde(x, A):
    """
    x: (N,3) 列是 (x,y,t)
    A: (N,2) 列是 (u,v)
    返回 [f_u, f_v]
    """
    u = A[:, 0:1]
    v = A[:, 1:2]
    # 时间导数
    u_t = dde.grad.jacobian(A, x, i=0, j=2)
    v_t = dde.grad.jacobian(A, x, i=1, j=2)
    # 空间 Laplacian
    u_xx = dde.grad.hessian(A, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(A, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(A, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(A, x, component=1, i=1, j=1)
    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy

    # 复 Ginzburg–Landau 残差
    # ∂A/∂t = A + (1 + i α) ∇²A − (1 + i β) |A|² A
    # real part:
    f_u = u_t - (
        u + lap_u - alpha * lap_v
    ) + (
        u * (u**2 + v**2) - beta * v * (u**2 + v**2)
    )
    # imag part:
    f_v = v_t - (
        v + lap_v + alpha * lap_u
    ) + (
        v * (u**2 + v**2) + beta * u * (u**2 + v**2)
    )
    return [f_u, f_v]

# 4. 定义初始条件 (t=0) 的 Dirichlet BC
def A0_u(x):
    # 举例：高斯包络初始实部
    return np.exp(-(x[:, 0:1]**2 + x[:, 1:2]**2))

def A0_v(x):
    # 初始虚部为零
    return np.zeros((x.shape[0], 1))

ic_u = dde.icbc.DirichletBC(
    geomtime, A0_u,
    lambda x, on_b: on_b and np.isclose(x[2], 0.0),
    component=0
)
ic_v = dde.icbc.DirichletBC(
    geomtime, A0_v,
    lambda x, on_b: on_b and np.isclose(x[2], 0.0),
    component=1
)


# 1) 明确L和T
L = 5.0   # 和你 geometry 定义保持一致
T = 1.0

# 2) 只在空间边界 (x,y)∈∂Ω，且排除 t=0 和 t=T 时面
def boundary_spatial(x, on_b):
    # x 是形如 [x, y, t] 的 1D array，on_b 是 bool
    spatial_edge = (abs(x[0]) >= L - 1e-6) or (abs(x[1]) >= L - 1e-6)
    temporal_inside = (x[2] > 0.0) and (x[2] < T)
    return on_b and spatial_edge and temporal_inside

# 3) 确保 BC 函数返回正确形状的数组
def zero_u(X):
    return np.zeros((X.shape[0], 1))
def zero_v(X):
    return np.zeros((X.shape[0], 1))

bc_u = dde.icbc.DirichletBC(
    geomtime, zero_u, boundary_spatial, component=0
)
bc_v = dde.icbc.DirichletBC(
    geomtime, zero_v, boundary_spatial, component=1
)

# 1）on_initial：只在 t≈0 时面生效  
def on_initial(x, on_b):
    # x 是一个长度 3 的向量 [x,y,t]
    return on_b and (abs(x[2]) < 1e-3)

# 2）向量化的初始值函数，输入 X.shape==(N,3)，返回 (N,1)
def A0_u(X):
    # 假设初始实部是高斯包络
    vals = np.exp(-(X[:,0]**2 + X[:,1]**2))
    return vals.reshape(-1,1)

def A0_v(X):
    # 初始虚部为零
    return np.zeros((X.shape[0],1))

# 3）用 DirichletBC 定义 IC
ic_u = dde.icbc.DirichletBC(
    geomtime,      # 依旧你的 GeometryXTime
    A0_u,          # 上面定义的向量化函数
    on_initial,    # 判断函数
    component=0    # u 分量
)
ic_v = dde.icbc.DirichletBC(
    geomtime,
    A0_v,
    on_initial,
    component=1    # v 分量
)



# 6. 构造训练数据：PDE + IC + BC
data = dde.data.TimePDE(
    geomtime,
    CGLE_pde,
    [ic_u, ic_v, bc_u, bc_v],
    num_domain=20000,
    num_initial=500,
    num_boundary=500,
)

# … 构造完 data 之后 …
# data.train_x 是所有采样点，列为 [x, y, t]
X_all = data.train_x

# 筛出初始条件点：t≈0
ic_idx = np.where(np.abs(X_all[:, 2]) < 1e-3)[0]
print("实际采到的 IC 点数：", len(ic_idx))

# 筛出空间边界点（排除 t=0 和 t=T）
bc_idx = np.where(
    ((np.abs(X_all[:,0]) >= L - 1e-6) | (np.abs(X_all[:,1]) >= L - 1e-6)) &
    (X_all[:,2] > 0) & (X_all[:,2] < T)
)[0]
print("实际采到的 BC 点数：", len(bc_idx))



# 7. 定义网络：6 层，每层 100 个神经元，tanh 激活
net = dde.nn.FNN([3] + [50] * 4 + [2], "sin", "Glorot uniform")




# 8. 建立模型并训练
model = dde.Model(data, net)

import numpy as np

# 随机生成一批 t≈0 的测试点，用来检测 IC 拟合是否达标
n_ic_test = 1000
# 在空间域 [-L,L]^2, t=0 采点
xs = np.random.uniform(-L, L, (n_ic_test, 1))
ys = np.random.uniform(-L, L, (n_ic_test, 1))
ts = np.zeros((n_ic_test, 1))
X_ic_test = np.hstack([xs, ys, ts])  # shape (n_ic_test, 3)

# 模型预测
pred_ic = model.predict(X_ic_test)  # shape (n_ic_test, 2) 列分别是 u_pred, v_pred

u_ic_pred = pred_ic[:, 0]  # (n_ic_test,)
v_ic_pred = pred_ic[:, 1]

# 计算真实初始条件值
u_ic_true = np.exp(-(xs.flatten()**2 + ys.flatten()**2))  # (n_ic_test,)
v_ic_true = np.zeros(n_ic_test)

print("IC 实部预测均值、最大、最小：", np.mean(u_ic_pred), np.max(u_ic_pred), np.min(u_ic_pred))
print("IC 实部真实均值、最大、最小：", np.mean(u_ic_true), np.max(u_ic_true), np.min(u_ic_true))
print("IC 虚部预测均值、最大、最小：", np.mean(v_ic_pred), np.max(v_ic_pred), np.min(v_ic_pred))

# 如果需要可视化散点对比
import matplotlib.pyplot as plt
plt.scatter(u_ic_true, u_ic_pred, s=5)
plt.xlabel("u_true at t=0")
plt.ylabel("u_pred at t=0")
plt.title("IC 实部：真实 vs 预测")
plt.show()


# 8.1 Adam 预训练
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(epochs=20000)

# 8.2 L-BFGS 精调
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 9. 取出反演参数
sess = model.sess
alpha_val = sess.run(alpha)
beta_val = sess.run(beta)
print(f"Inferred alpha = {alpha_val:.5f}")
print(f"Inferred beta  = {beta_val:.5f}")

# 10. 可视化损失与场预测
# 1. 在 [−L,L]^2×[0,T] 里随机采 n_test 个测试点
n_test = 5000
X_test = geomtime.random_points(n_test)  # shape (n_test,3)

# 2. 网络预测
y_pred = model.predict(X_test)            # shape (n_test,2), 列为 [u_pred, v_pred]

# 3. 保存到文件
data = np.hstack([X_test, y_pred])
# 列名： x, y, t, u_pred, v_pred
np.savetxt("cgle_predictions.dat", data,
           header="x y t u_pred v_pred")

# 4. （可选）在某个固定时刻 t0 可视化 u 的分布
t0 = T  # 或者 0.5*T
# 构造规则网格
nx = ny = 200
xs = np.linspace(-L, L, nx)
ys = np.linspace(-L, L, ny)
xx, yy = np.meshgrid(xs, ys)
tt = np.full_like(xx, t0)
Xg = np.vstack([xx.ravel(), yy.ravel(), tt.ravel()]).T
pred = model.predict(Xg)        # 返回 shape=(nx*ny, 2)
Ug = pred[:, 0].reshape(nx, ny) # 实部
Vg = pred[:, 1].reshape(nx, ny) # 虚部

# 重塑并画图
Ug = Ug.reshape(nx, ny)
plt.figure(figsize=(5,4))
plt.pcolormesh(xs, ys, Ug, shading="auto")
plt.colorbar(label="u(x,y,t0)")
plt.title(f"u at t={t0:.2f}")
plt.savefig("u_t0.png", dpi=150)
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
alpha = alpha_val
beta = beta_val

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
