# 线性代数知识文档 — 人脸追踪项目

## 📖 项目概述

本项目 **RETRO FACE TRACKER** 是一个基于 Python + OpenCV 的实时人脸追踪系统，融合了多个核心线性代数概念。项目使用摄像头捕获实时视频流，通过 Haar 级联分类器检测人脸，并利用**卡尔曼滤波器 (Kalman Filter)** 实现平滑追踪；同时运用 **SVD（奇异值分解）** 实现实时视频流的图像压缩（故障艺术效果）。

---

## 一、卡尔曼滤波器 (Kalman Filter)

### 1.1 什么是卡尔曼滤波？

卡尔曼滤波是一种**最优线性状态估计算法**，由 Rudolf Kálmán 于 1960 年提出。它通过「预测-更新」两个循环步骤，结合系统的数学模型和带有噪声的观测数据，递归地估计系统的状态。

在本项目中，卡尔曼滤波器用于：
- 平滑人脸追踪框的位置（减少检测器的抖动）
- 在人脸短暂消失时预测人脸的位置

### 1.2 状态向量 (State Vector)

我们定义状态向量 **X** 为一个 4×1 的列向量：

$$
\mathbf{X} = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}
$$

其中：
- $(x, y)$ — 人脸中心在图像中的像素坐标（位置）
- $(v_x, v_y)$ — 人脸中心的运动速度（每帧像素变化量）

### 1.3 状态转移矩阵 (State Transition Matrix)

状态转移矩阵 **F** 描述了从时刻 $t$ 到 $t+1$ 的线性变换关系。我们假设人脸做**匀速直线运动**：

$$
\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

其物理意义为：
$$
x_{t+1} = x_t + v_x \cdot \Delta t, \quad y_{t+1} = y_t + v_y \cdot \Delta t
$$

这是一个**线性变换矩阵**，将当前状态通过矩阵-向量乘法映射到预测的下一个状态：
$$
\mathbf{X}_{pred} = \mathbf{F} \cdot \mathbf{X}
$$

### 1.4 观测矩阵 (Observation Matrix)

观测矩阵 **H** 将状态空间映射（投影）到观测空间：

$$
\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

我们只能直接观测人脸的位置 $(x, y)$，而不能直接观测速度 $(v_x, v_y)$，所以 H 矩阵的后两列为零。H 实现了从 4 维空间到 2 维空间的**线性投影**。

### 1.5 协方差矩阵 (Covariance Matrix)

协方差矩阵 **P** 是一个 4×4 的**对称正定矩阵**，量化了状态估计的不确定性：

$$
\mathbf{P} = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xv_x} & \sigma_{xv_y} \\ \sigma_{yx} & \sigma_{yy} & \sigma_{yv_x} & \sigma_{yv_y} \\ \sigma_{v_xx} & \sigma_{v_xy} & \sigma_{v_xv_x} & \sigma_{v_xv_y} \\ \sigma_{v_yx} & \sigma_{v_yy} & \sigma_{v_yv_x} & \sigma_{v_yv_y} \end{bmatrix}
$$

**性质：**
- **对称性**：$P = P^T$（协方差矩阵总是对称的）
- **正定性**：对任意非零向量 $\mathbf{v}$，$\mathbf{v}^T \mathbf{P} \mathbf{v} > 0$
- **对角线元素**：表示各状态分量的方差（不确定性）
- **非对角线元素**：表示状态分量之间的相关性

### 1.6 预测步骤 (Prediction Step)

$$
\mathbf{X}_{pred} = \mathbf{F} \cdot \mathbf{X}
$$
$$
\mathbf{P}_{pred} = \mathbf{F} \cdot \mathbf{P} \cdot \mathbf{F}^T + \mathbf{Q}
$$

涉及的线性代数运算：
- **矩阵-向量乘法**：$\mathbf{F} \cdot \mathbf{X}$
- **矩阵转置**：$\mathbf{F}^T$
- **二次型变换**：$\mathbf{F} \cdot \mathbf{P} \cdot \mathbf{F}^T$（保持 P 的对称正定性）
- **矩阵加法**：$+ \mathbf{Q}$（加入过程噪声）

### 1.7 更新步骤 (Update Step)

当检测到人脸时，我们用观测值来修正预测：

$$
\mathbf{y} = \mathbf{z} - \mathbf{H} \cdot \mathbf{X}_{pred} \quad \text{(新息/残差)}
$$
$$
\mathbf{S} = \mathbf{H} \cdot \mathbf{P}_{pred} \cdot \mathbf{H}^T + \mathbf{R} \quad \text{(新息协方差)}
$$
$$
\mathbf{K} = \mathbf{P}_{pred} \cdot \mathbf{H}^T \cdot \mathbf{S}^{-1} \quad \text{(卡尔曼增益)}
$$
$$
\mathbf{X} = \mathbf{X}_{pred} + \mathbf{K} \cdot \mathbf{y} \quad \text{(状态更新)}
$$
$$
\mathbf{P} = (\mathbf{I} - \mathbf{K} \cdot \mathbf{H}) \cdot \mathbf{P}_{pred} \quad \text{(协方差更新)}
$$

涉及的关键线性代数运算：
- **矩阵求逆**：$\mathbf{S}^{-1}$（计算 2×2 矩阵的逆）
- **单位矩阵**：$\mathbf{I}$（4×4 单位矩阵）
- **卡尔曼增益 K**：在预测和观测之间找到最优的加权平衡

---

## 二、奇异值分解 (SVD — Singular Value Decomposition)

### 2.1 SVD 分解的定义

对于任意 $m \times n$ 的实数矩阵 $\mathbf{A}$，都存在如下分解：

$$
\mathbf{A} = \mathbf{U} \cdot \mathbf{\Sigma} \cdot \mathbf{V}^T
$$

其中：
- $\mathbf{U}$：$m \times m$ 的**正交矩阵**（$\mathbf{U}^T \mathbf{U} = \mathbf{I}$），列向量称为**左奇异向量**
- $\mathbf{\Sigma}$：$m \times n$ 的**对角矩阵**，对角线元素 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ 称为**奇异值**
- $\mathbf{V}^T$：$n \times n$ 的**正交矩阵**，行向量称为**右奇异向量**

### 2.2 图像压缩原理

一张灰度图像可以看作一个 $m \times n$ 的矩阵（每个元素是一个像素值）。对其进行 SVD 分解后，只保留前 $k$ 个最大的奇异值，即可得到原图的**最优低秩近似**：

$$
\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \cdot \mathbf{u}_i \cdot \mathbf{v}_i^T
$$

这等价于：
$$
\mathbf{A}_k = \mathbf{U}_{[:, :k]} \cdot \mathbf{\Sigma}_{[:k, :k]} \cdot \mathbf{V}^T_{[:k, :]}
$$

### 2.3 Eckart-Young 定理

SVD 低秩近似之所以是"最优"的，由 **Eckart-Young 定理** 保证：

> 在所有秩为 $k$ 的矩阵中，$\mathbf{A}_k$ 是在 Frobenius 范数意义下与原矩阵 $\mathbf{A}$ 最接近的矩阵。

$$
\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}
$$

### 2.4 在本项目中的应用

本项目对实时视频帧的每个颜色通道（B、G、R）分别进行 SVD 分解：
- **k 值大**（如 100-200）：画面接近原始效果
- **k 值中等**（如 20-50）：画面呈现朦胧的艺术效果
- **k 值很小**（如 1-10）：画面呈现强烈的"故障"（glitch）效果

压缩比 = 原始存储量 / 压缩后存储量：
- 原始：$m \times n$ 个像素值
- 压缩后：$k \times (m + n + 1)$ 个值
- 当 $k \ll \min(m, n)$ 时，压缩效果显著

### 2.5 正交矩阵的性质

SVD 中的 $\mathbf{U}$ 和 $\mathbf{V}$ 都是正交矩阵，满足：
- $\mathbf{U}^T \mathbf{U} = \mathbf{I}$（列向量两两正交且为单位向量）
- $\mathbf{V}^T \mathbf{V} = \mathbf{I}$
- 正交矩阵的逆等于它的转置：$\mathbf{U}^{-1} = \mathbf{U}^T$
- 正交变换保持向量的长度和角度不变

---

## 三、其他涉及的线性代数概念

### 3.1 矩阵乘法 (Matrix Multiplication)

本项目大量使用矩阵乘法，如卡尔曼滤波中的 $\mathbf{F} \cdot \mathbf{X}$、$\mathbf{H} \cdot \mathbf{P}$ 等。矩阵乘法的定义：

$$
(\mathbf{AB})_{ij} = \sum_{k} a_{ik} \cdot b_{kj}
$$

在 Python 中使用 `@` 运算符或 `np.dot()` 实现。

### 3.2 矩阵转置 (Matrix Transpose)

矩阵转置 $\mathbf{A}^T$ 将矩阵的行和列互换：
$$
(\mathbf{A}^T)_{ij} = \mathbf{A}_{ji}
$$

在卡尔曼滤波的协方差传播中广泛使用，如 $\mathbf{F} \cdot \mathbf{P} \cdot \mathbf{F}^T$。

### 3.3 矩阵求逆 (Matrix Inverse)

方阵 $\mathbf{A}$ 的逆矩阵 $\mathbf{A}^{-1}$ 满足：
$$
\mathbf{A} \cdot \mathbf{A}^{-1} = \mathbf{A}^{-1} \cdot \mathbf{A} = \mathbf{I}
$$

在卡尔曼增益的计算中，需要对新息协方差矩阵 $\mathbf{S}$ 求逆。

### 3.4 单位矩阵 (Identity Matrix)

单位矩阵 $\mathbf{I}$ 是对角线为 1、其余为 0 的方阵，满足：
$$
\mathbf{I} \cdot \mathbf{A} = \mathbf{A} \cdot \mathbf{I} = \mathbf{A}
$$

在协方差更新公式 $\mathbf{P} = (\mathbf{I} - \mathbf{K} \cdot \mathbf{H}) \cdot \mathbf{P}$ 中使用。

### 3.5 特征值与特征向量 (Eigenvalues & Eigenvectors)

奇异值与特征值密切相关：
- $\mathbf{A}^T \mathbf{A}$ 的特征值的平方根就是 $\mathbf{A}$ 的奇异值
- 协方差矩阵 $\mathbf{P}$ 的特征值反映了不确定性的主要方向和大小

---

## 四、项目中线性代数知识的对应关系

| 线性代数概念 | 项目中的应用 | 代码位置 |
|---|---|---|
| 状态向量 (State Vector) | 人脸位置和速度的表示 | `KalmanFilter2D.__init__` |
| 状态转移矩阵 (Transition Matrix) | 匀速运动模型 | `KalmanFilter2D.__init__` |
| 观测矩阵 (Observation Matrix) | 位置提取（4D → 2D 投影） | `KalmanFilter2D.__init__` |
| 协方差矩阵 (Covariance Matrix) | 预测不确定性的量化 | `KalmanFilter2D.predict/update` |
| 矩阵求逆 (Matrix Inverse) | 卡尔曼增益的计算 | `KalmanFilter2D.update` |
| 矩阵转置 (Matrix Transpose) | 协方差传播 | `KalmanFilter2D.predict/update` |
| SVD 分解 | 图像压缩 / 故障艺术 | `apply_svd_compression` |
| 低秩近似 (Low-rank Approx.) | 保留 k 个奇异值 | `apply_svd_compression` |
| 正交矩阵 (Orthogonal Matrix) | SVD 中的 U 和 V | `apply_svd_compression` |
| 对角矩阵 (Diagonal Matrix) | SVD 中的 Σ 矩阵 | `apply_svd_compression` |

---

## 五、参考资料

1. Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering.
2. Strang, G. (2016). *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press.
3. Eckart, C., & Young, G. (1936). *The Approximation of One Matrix by Another of Lower Rank*. Psychometrika.
4. OpenCV Documentation: https://docs.opencv.org/
5. NumPy Linear Algebra: https://numpy.org/doc/stable/reference/routines.linalg.html

---

> 📝 本文档是项目的线性代数知识说明，旨在帮助理解项目中涉及的数学原理和它们在代码中的具体实现方式。
