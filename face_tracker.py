"""
================================================================================
  ██████╗ ███████╗████████╗██████╗  ██████╗     ███████╗ █████╗  ██████╗███████╗
  ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗    ██╔════╝██╔══██╗██╔════╝██╔════╝
  ██████╔╝█████╗     ██║   ██████╔╝██║   ██║    █████╗  ███████║██║     █████╗
  ██╔══██╗██╔══╝     ██║   ██╔══██╗██║   ██║    ██╔══╝  ██╔══██║██║     ██╔══╝
  ██║  ██║███████╗   ██║   ██║  ██║╚██████╔╝    ██║     ██║  ██║╚██████╗███████╗
  ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝
  RETRO FACE TRACKER — 基于线性代数的人脸追踪系统
================================================================================

项目简介 / Project Overview:
    本项目利用 OpenCV 的 Haar 级联分类器进行人脸检测，结合卡尔曼滤波器
    (Kalman Filter) 实现平滑的人脸追踪。同时运用 SVD（奇异值分解）对
    实时视频流进行图像压缩，展示线性代数在计算机视觉中的实际应用。

    UI 采用 70-80 年代科幻复古终端风格，灵感来源于战斗机 HUD 界面。

核心线性代数知识:
    1. 卡尔曼滤波 (Kalman Filter) — 状态预测与更新中的矩阵运算
    2. SVD 分解 (Singular Value Decomposition) — 图像压缩与降维
    3. 协方差矩阵 (Covariance Matrix) — 量化预测不确定性

操作说明 / Controls:
    [S] — 切换 SVD 故障艺术模式 / Toggle SVD Glitch Art Mode
    [+] / [-] — 增加/减少 SVD 保留的奇异值数量
    [M] — 切换 纯检测 / 卡尔曼追踪 模式
    [C] — 生成对比图表
    [Q] / [ESC] — 退出程序

作者: 学生项目
日期: 2026
"""

import cv2
import numpy as np
import matplotlib

# ============================================================
# 使用非交互式后端，避免 matplotlib 与 OpenCV 窗口冲突
# ============================================================
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os


# ============================================================
# 第一部分: 卡尔曼滤波器类
# Part 1: Kalman Filter Class
# ============================================================
# 卡尔曼滤波器是一种最优线性状态估计器，核心是通过"预测-更新"
# 两个步骤，利用矩阵运算不断修正对目标状态的估计。
# ============================================================

class KalmanFilter2D:
    """
    二维卡尔曼滤波器，用于追踪人脸在画面中的位置。

    状态向量 X = [x, y, vx, vy]^T
        其中 (x, y) 是人脸中心坐标，(vx, vy) 是速度分量。

    线性代数知识点:
        - 状态转移矩阵 F: 描述系统从 t 时刻到 t+1 时刻的线性变换
        - 观测矩阵 H: 将状态空间映射到观测空间的线性变换
        - 协方差矩阵 P: 对称正定矩阵，量化状态估计的不确定性
        - 卡尔曼增益 K: 由矩阵求逆运算得到，平衡预测与观测的权重
    """

    def __init__(self, dt=1.0):
        """
        初始化卡尔曼滤波器。

        参数:
            dt (float): 时间步长，默认为 1（即每帧为一个时间单位）
        """
        # ----------------------------------------------------------
        # 状态向量 X: 4×1 矩阵 (列向量)
        # X = [x, y, vx, vy]^T
        # 初始化为零向量，表示初始位置和速度均未知
        # ----------------------------------------------------------
        self.X = np.zeros((4, 1), dtype=np.float64)

        # ----------------------------------------------------------
        # 状态转移矩阵 F: 4×4 矩阵
        # 描述匀速运动模型: x_new = x + vx * dt
        #                    y_new = y + vy * dt
        # F = [[1, 0, dt, 0 ],
        #      [0, 1, 0,  dt],
        #      [0, 0, 1,  0 ],
        #      [0, 0, 0,  1 ]]
        # 这是一个线性变换矩阵，将当前状态映射到下一时刻的状态
        # ----------------------------------------------------------
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float64)

        # ----------------------------------------------------------
        # 观测矩阵 H: 2×4 矩阵
        # 我们只能观测到位置 (x, y)，不能直接观测速度
        # H 将 4 维状态空间投影到 2 维观测空间
        # H = [[1, 0, 0, 0],
        #      [0, 1, 0, 0]]
        # ----------------------------------------------------------
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # ----------------------------------------------------------
        # 协方差矩阵 P: 4×4 对称正定矩阵
        # 量化状态估计的不确定性
        # 初始设为较大值，表示初始状态非常不确定
        # P 是对称矩阵: P = P^T，且对角线元素为各状态分量的方差
        # ----------------------------------------------------------
        self.P = np.eye(4, dtype=np.float64) * 1000.0

        # ----------------------------------------------------------
        # 过程噪声协方差矩阵 Q: 4×4 矩阵
        # 模拟系统模型中的不确定性（如人脸的加速度变化）
        # 较小的 Q 意味着我们更信任运动模型
        # ----------------------------------------------------------
        self.Q = np.eye(4, dtype=np.float64) * 0.1
        self.Q[0, 0] = 0.5   # 位置 x 的噪声
        self.Q[1, 1] = 0.5   # 位置 y 的噪声
        self.Q[2, 2] = 1.0   # 速度 vx 的噪声
        self.Q[3, 3] = 1.0   # 速度 vy 的噪声

        # ----------------------------------------------------------
        # 观测噪声协方差矩阵 R: 2×2 矩阵
        # 模拟人脸检测的测量噪声（检测框位置的抖动）
        # ----------------------------------------------------------
        self.R = np.eye(2, dtype=np.float64) * 5.0

        # ----------------------------------------------------------
        # 记录是否已经初始化（第一次检测到人脸时初始化状态）
        # ----------------------------------------------------------
        self.initialized = False

    def predict(self):
        """
        预测步骤 — 卡尔曼滤波的第一阶段

        数学公式:
            X_pred = F · X          (状态预测: 矩阵-向量乘法)
            P_pred = F · P · F^T + Q  (协方差预测: 矩阵乘法与转置)

        线性代数知识:
            - F · X 是矩阵与向量的乘法，实现线性变换
            - F · P · F^T 是二次型变换，保持 P 的对称正定性
            - F^T 是 F 的转置矩阵
        """
        # 状态预测: X_pred = F · X
        self.X = self.F @ self.X

        # 协方差预测: P_pred = F · P · F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.X

    def update(self, z):
        """
        更新步骤 — 卡尔曼滤波的第二阶段

        参数:
            z: 观测向量 [x_obs, y_obs]^T，即检测到的人脸中心坐标

        数学公式:
            y = z - H · X           (新息/残差: 观测与预测的差)
            S = H · P · H^T + R     (新息协方差矩阵)
            K = P · H^T · S^(-1)    (卡尔曼增益: 需要矩阵求逆)
            X = X + K · y           (状态更新)
            P = (I - K · H) · P     (协方差更新)

        线性代数知识:
            - S^(-1) 是矩阵求逆运算，这里是 2×2 矩阵的逆
            - K (卡尔曼增益) 决定了预测和观测之间的最优权重
            - I 是单位矩阵
        """
        # 将观测值转为列向量
        z = np.array(z, dtype=np.float64).reshape(2, 1)

        # 计算新息（残差）: y = z - H · X
        y = z - self.H @ self.X

        # 计算新息协方差: S = H · P · H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        # 计算卡尔曼增益: K = P · H^T · S^(-1)
        # np.linalg.inv(S) 执行矩阵求逆运算
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新: X = X + K · y
        self.X = self.X + K @ y

        # 协方差更新: P = (I - K · H) · P
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.X

    def init_state(self, x, y):
        """
        使用第一次检测到的人脸位置初始化状态向量。

        参数:
            x, y: 人脸中心的初始坐标
        """
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float64)
        self.initialized = True


# ============================================================
# 第二部分: SVD 图像压缩
# Part 2: SVD Image Compression (Glitch Art)
# ============================================================
# SVD（奇异值分解）将任意 m×n 矩阵 A 分解为:
#     A = U · Σ · V^T
# 其中:
#     U: m×m 正交矩阵（左奇异向量）
#     Σ: m×n 对角矩阵（奇异值，按降序排列）
#     V^T: n×n 正交矩阵（右奇异向量的转置）
#
# 通过只保留前 k 个最大的奇异值，可以实现图像的低秩近似，
# 这就是 SVD 图像压缩的核心原理。
# ============================================================

def apply_svd_compression(frame, k):
    """
    对图像帧应用 SVD 压缩（故障艺术效果）。

    参数:
        frame: 输入图像 (BGR 格式的 numpy 数组)
        k: 保留的奇异值数量。k 越小，压缩越强，图像越"故障"

    返回:
        compressed: 压缩后的图像

    数学原理:
        对图像的每个颜色通道（B, G, R），分别执行 SVD 分解:
            Channel = U · diag(σ₁, σ₂, ..., σₙ) · V^T

        只保留前 k 个奇异值:
            Channel_approx = U[:, :k] · diag(σ₁, ..., σₖ) · V[:k, :]

        这是矩阵的最优低秩近似（Eckart-Young 定理）。
    """
    compressed_channels = []

    # 对 B, G, R 三个通道分别进行 SVD 分解
    for i in range(3):
        # 提取单个颜色通道（这是一个 m×n 的实数矩阵）
        channel = frame[:, :, i].astype(np.float64)

        # ----------------------------------------------------------
        # 执行 SVD 分解: Channel = U · Σ · V^T
        # U: 左奇异向量矩阵 (m×m)
        # s: 奇异值向量 (min(m,n) 个值，降序排列)
        # Vt: 右奇异向量矩阵的转置 (n×n)
        # full_matrices=False 返回经济型 SVD，节省内存
        # ----------------------------------------------------------
        U, s, Vt = np.linalg.svd(channel, full_matrices=False)

        # 限制 k 不超过奇异值的总数
        k_actual = min(k, len(s))

        # ----------------------------------------------------------
        # 低秩近似: 只保留前 k 个奇异值
        # Channel_approx = U[:, :k] · diag(σ₁, ..., σₖ) · V[:k, :]
        # 这里 np.diag(s[:k_actual]) 构造对角矩阵
        # ----------------------------------------------------------
        compressed = U[:, :k_actual] @ np.diag(s[:k_actual]) @ Vt[:k_actual, :]

        # 将像素值裁剪到 [0, 255] 范围并转为 uint8
        compressed = np.clip(compressed, 0, 255).astype(np.uint8)
        compressed_channels.append(compressed)

    # 合并三个通道为 BGR 图像
    return cv2.merge(compressed_channels)


# ============================================================
# 第三部分: 复古终端 UI 绘制
# Part 3: Retro Terminal UI Rendering
# ============================================================
# UI 灵感来源于 70-80 年代战斗机 HUD (抬头显示器) 界面
# 使用绿色/琥珀色调的线条和文字，模拟 CRT 显示效果
# ============================================================

# 定义复古配色方案
COLOR_GREEN = (0, 255, 0)          # 主色: 荧光绿 (HUD 经典色)
COLOR_DARK_GREEN = (0, 180, 0)     # 辅色: 暗绿
COLOR_AMBER = (0, 200, 255)        # 琥珀色 (另一种经典终端色)
COLOR_RED = (0, 0, 255)            # 警告色: 红色
COLOR_CYAN = (255, 255, 0)         # 青色: 用于高亮信息
COLOR_DIM_GREEN = (0, 100, 0)      # 暗淡绿: 用于背景装饰
COLOR_BG_OVERLAY = (0, 0, 0)       # 黑色遮罩


def draw_scanlines(frame, intensity=30):
    """
    绘制 CRT 扫描线效果，模拟老式显示器的显示特征。

    参数:
        frame: 输入图像
        intensity: 扫描线的暗度 (0-255)
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # 每隔 2 行绘制一条暗线，模拟 CRT 扫描线
    for y in range(0, h, 2):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    # 使用透明度混合
    cv2.addWeighted(overlay, intensity / 255.0, frame, 1 - intensity / 255.0, 0, frame)


def draw_hud_border(frame):
    """
    绘制 HUD 风格的边框，模拟战斗机座舱显示器。
    使用多层边框和角落装饰营造科幻感。
    """
    h, w = frame.shape[:2]

    # 外层边框
    cv2.rectangle(frame, (5, 5), (w - 5, h - 5), COLOR_DIM_GREEN, 1)
    # 内层边框
    cv2.rectangle(frame, (15, 15), (w - 15, h - 15), COLOR_GREEN, 1)

    corner_len = 30  # 角落装饰线的长度

    # 四个角落的 L 形装饰
    # 左上角
    cv2.line(frame, (15, 15), (15 + corner_len, 15), COLOR_AMBER, 2)
    cv2.line(frame, (15, 15), (15, 15 + corner_len), COLOR_AMBER, 2)
    # 右上角
    cv2.line(frame, (w - 15, 15), (w - 15 - corner_len, 15), COLOR_AMBER, 2)
    cv2.line(frame, (w - 15, 15), (w - 15, 15 + corner_len), COLOR_AMBER, 2)
    # 左下角
    cv2.line(frame, (15, h - 15), (15 + corner_len, h - 15), COLOR_AMBER, 2)
    cv2.line(frame, (15, h - 15), (15, h - 15 - corner_len), COLOR_AMBER, 2)
    # 右下角
    cv2.line(frame, (w - 15, h - 15), (w - 15 - corner_len, h - 15), COLOR_AMBER, 2)
    cv2.line(frame, (w - 15, h - 15), (w - 15, h - 15 - corner_len), COLOR_AMBER, 2)

    # 顶部标题栏
    cv2.rectangle(frame, (20, 20), (w - 20, 55), COLOR_BG_OVERLAY, -1)
    cv2.rectangle(frame, (20, 20), (w - 20, 55), COLOR_GREEN, 1)
    cv2.putText(frame, "[ RETRO FACE TRACKER v1.0 ]", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 1)

    # 显示当前时间戳 (模拟军用时间格式)
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, f"SYS TIME: {timestamp}", (w - 230, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AMBER, 1)


def draw_crosshair(frame, cx, cy, size=40, color=COLOR_GREEN):
    """
    在目标位置绘制十字准星（战斗机 HUD 风格）。

    参数:
        frame: 输入图像
        cx, cy: 准星中心坐标
        size: 准星大小
        color: 准星颜色
    """
    half = size // 2
    gap = 8  # 中心间隙

    # 绘制带间隙的十字线
    cv2.line(frame, (cx - half, cy), (cx - gap, cy), color, 1)  # 左
    cv2.line(frame, (cx + gap, cy), (cx + half, cy), color, 1)  # 右
    cv2.line(frame, (cx, cy - half), (cx, cy - gap), color, 1)  # 上
    cv2.line(frame, (cx, cy + gap), (cx, cy + half), color, 1)  # 下

    # 绘制角落的小方块
    bracket = 10
    cv2.line(frame, (cx - half, cy - half), (cx - half + bracket, cy - half), color, 1)
    cv2.line(frame, (cx - half, cy - half), (cx - half, cy - half + bracket), color, 1)
    cv2.line(frame, (cx + half, cy - half), (cx + half - bracket, cy - half), color, 1)
    cv2.line(frame, (cx + half, cy - half), (cx + half, cy - half + bracket), color, 1)
    cv2.line(frame, (cx - half, cy + half), (cx - half + bracket, cy + half), color, 1)
    cv2.line(frame, (cx - half, cy + half), (cx - half, cy + half - bracket), color, 1)
    cv2.line(frame, (cx + half, cy + half), (cx + half - bracket, cy + half), color, 1)
    cv2.line(frame, (cx + half, cy + half), (cx + half, cy + half - bracket), color, 1)


def draw_tracking_box(frame, x, y, w, h, label="TARGET", color=COLOR_GREEN):
    """
    绘制复古风格的追踪框。

    参数:
        frame: 输入图像
        x, y, w, h: 追踪框的位置和大小
        label: 标签文字
        color: 框的颜色
    """
    # 绘制主追踪框（虚线效果通过角落装饰实现）
    corner = 15

    # 四个角的 L 形线段
    # 左上
    cv2.line(frame, (x, y), (x + corner, y), color, 2)
    cv2.line(frame, (x, y), (x, y + corner), color, 2)
    # 右上
    cv2.line(frame, (x + w, y), (x + w - corner, y), color, 2)
    cv2.line(frame, (x + w, y), (x + w, y + corner), color, 2)
    # 左下
    cv2.line(frame, (x, y + h), (x + corner, y + h), color, 2)
    cv2.line(frame, (x, y + h), (x, y + h - corner), color, 2)
    # 右下
    cv2.line(frame, (x + w, y + h), (x + w - corner, y + h), color, 2)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner), color, 2)

    # 绘制标签背景
    label_w = len(label) * 10 + 20
    cv2.rectangle(frame, (x, y - 22), (x + label_w, y - 2), COLOR_BG_OVERLAY, -1)
    cv2.rectangle(frame, (x, y - 22), (x + label_w, y - 2), color, 1)
    cv2.putText(frame, label, (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 绘制中心十字准星
    cx, cy = x + w // 2, y + h // 2
    draw_crosshair(frame, cx, cy, size=min(w, h) // 2, color=color)


def draw_matrix_display(frame, kalman, x_offset, y_offset):
    """
    在屏幕上实时显示卡尔曼滤波器的状态向量 X 和协方差矩阵 P。
    这是"可视化看不见的矩阵"功能的核心。

    参数:
        frame: 输入图像
        kalman: KalmanFilter2D 实例
        x_offset, y_offset: 显示位置
    """
    h, w = frame.shape[:2]

    # ----------------------------------------------------------
    # 绘制状态向量 X 的可视化面板
    # X = [x, y, vx, vy]^T
    # ----------------------------------------------------------
    panel_w, panel_h = 250, 180
    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset, y_offset),
                  (x_offset + panel_w, y_offset + panel_h),
                  COLOR_BG_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 面板边框
    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + panel_w, y_offset + panel_h),
                  COLOR_GREEN, 1)

    # 标题
    cv2.putText(frame, "[ STATE VECTOR X ]", (x_offset + 10, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_AMBER, 1)

    # 显示状态向量的每个分量
    labels = ["POS_X", "POS_Y", "VEL_X", "VEL_Y"]
    for i, label in enumerate(labels):
        val = kalman.X[i, 0]
        y_pos = y_offset + 45 + i * 22
        cv2.putText(frame, f"{label}: {val:8.2f}",
                    (x_offset + 15, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1)

        # 绘制小型条形图表示数值大小
        bar_len = int(min(abs(val) / 5.0, 1.0) * 80)
        bar_color = COLOR_GREEN if val >= 0 else COLOR_RED
        cv2.rectangle(frame, (x_offset + 155, y_pos - 10),
                      (x_offset + 155 + bar_len, y_pos - 2),
                      bar_color, -1)

    # ----------------------------------------------------------
    # 绘制协方差矩阵 P 的热力图可视化
    # P 是 4×4 对称正定矩阵，对角线元素表示各分量的方差
    # ----------------------------------------------------------
    cv2.putText(frame, "[ COVARIANCE P ]", (x_offset + 10, y_offset + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_AMBER, 1)

    # 将 P 矩阵的对角线元素可视化为色块
    diag_vals = np.diag(kalman.P)
    max_val = max(np.max(np.abs(diag_vals)), 1.0)
    for i in range(4):
        intensity = min(int(abs(diag_vals[i]) / max_val * 255), 255)
        color = (0, intensity, 0)
        bx = x_offset + 15 + i * 55
        by = y_offset + 150
        cv2.rectangle(frame, (bx, by), (bx + 45, by + 20), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + 45, by + 20), COLOR_GREEN, 1)
        cv2.putText(frame, f"{diag_vals[i]:.1f}", (bx + 2, by + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_AMBER, 1)


def draw_status_bar(frame, mode, svd_mode, svd_k, fps, face_detected):
    """
    绘制底部状态栏，显示当前模式和控制提示。

    参数:
        frame: 输入图像
        mode: 当前运行模式 ("DETECT" 或 "KALMAN")
        svd_mode: SVD 模式是否开启
        svd_k: 保留的奇异值数量
        fps: 当前帧率
        face_detected: 是否检测到人脸
    """
    h, w = frame.shape[:2]
    bar_y = h - 65

    # 底部状态栏背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, bar_y), (w - 15, h - 15), COLOR_BG_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, bar_y), (w - 15, h - 15), COLOR_GREEN, 1)

    # 模式指示
    mode_color = COLOR_GREEN if mode == "KALMAN" else COLOR_AMBER
    cv2.putText(frame, f"MODE: {mode}", (25, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1)

    # SVD 状态
    svd_text = f"SVD: ON (k={svd_k})" if svd_mode else "SVD: OFF"
    svd_color = COLOR_CYAN if svd_mode else COLOR_DIM_GREEN
    cv2.putText(frame, svd_text, (180, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, svd_color, 1)

    # FPS 显示
    cv2.putText(frame, f"FPS: {fps:.1f}", (380, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    # 人脸检测状态
    status = "LOCKED" if face_detected else "SCANNING..."
    status_color = COLOR_GREEN if face_detected else COLOR_RED
    cv2.putText(frame, f"TARGET: {status}", (480, bar_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

    # 控制提示
    controls = "[S]SVD  [+/-]K  [M]MODE  [C]CHART  [Q]QUIT"
    cv2.putText(frame, controls, (25, bar_y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_DIM_GREEN, 1)


def draw_radar_decoration(frame, cx, cy, angle):
    """
    绘制雷达扫描动画装饰（纯视觉效果，增强科幻感）。

    参数:
        frame: 输入图像
        cx, cy: 雷达中心坐标
        angle: 当前扫描角度
    """
    radius = 40

    # 同心圆
    cv2.circle(frame, (cx, cy), radius, COLOR_DIM_GREEN, 1)
    cv2.circle(frame, (cx, cy), radius // 2, COLOR_DIM_GREEN, 1)

    # 十字线
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), COLOR_DIM_GREEN, 1)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), COLOR_DIM_GREEN, 1)

    # 扫描线（旋转的半径线）
    end_x = int(cx + radius * np.cos(np.radians(angle)))
    end_y = int(cy + radius * np.sin(np.radians(angle)))
    cv2.line(frame, (cx, cy), (end_x, end_y), COLOR_GREEN, 1)


# ============================================================
# 第四部分: 对比实验与图表生成
# Part 4: Comparison Experiment & Chart Generation
# ============================================================
# 对比"纯检测"和"检测+卡尔曼滤波"两种模式的追踪效果，
# 用 matplotlib 绘制位置变化曲线。
# ============================================================

def generate_comparison_chart(detect_positions, kalman_positions, output_path="comparison_chart.png"):
    """
    生成对比图表，展示纯检测 vs 卡尔曼滤波追踪的位置变化曲线。

    参数:
        detect_positions: 纯检测模式下的位置记录 [(x, y), ...]
        kalman_positions: 卡尔曼滤波模式下的位置记录 [(x, y), ...]
        output_path: 图表保存路径

    图表说明:
        - 上图: X 坐标随时间的变化
        - 下图: Y 坐标随时间的变化
        可以明显看到卡尔曼滤波使曲线更平滑，减少了抖动
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='#00ff00')
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['top'].set_color('#004400')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['right'].set_color('#004400')
        ax.grid(True, color='#003300', linestyle='--', alpha=0.5)

    # 提取 x, y 坐标序列
    if detect_positions:
        det_x = [p[0] for p in detect_positions]
        det_y = [p[1] for p in detect_positions]
        det_frames = range(len(detect_positions))
    else:
        det_x, det_y, det_frames = [], [], []

    if kalman_positions:
        kal_x = [p[0] for p in kalman_positions]
        kal_y = [p[1] for p in kalman_positions]
        kal_frames = range(len(kalman_positions))
    else:
        kal_x, kal_y, kal_frames = [], [], []

    # X 坐标对比图
    axes[0].set_title('X Position Over Time', color='#00ff00', fontsize=14, fontfamily='monospace')
    if det_x:
        axes[0].plot(det_frames, det_x, color='#ff6600', linewidth=1, alpha=0.7, label='Detection Only')
    if kal_x:
        axes[0].plot(kal_frames, kal_x, color='#00ff00', linewidth=1.5, label='Kalman Filter')
    axes[0].set_ylabel('X (pixels)', color='#00ff00', fontfamily='monospace')
    axes[0].legend(facecolor='#0a0a0a', edgecolor='#00ff00', labelcolor='#00ff00')

    # Y 坐标对比图
    axes[1].set_title('Y Position Over Time', color='#00ff00', fontsize=14, fontfamily='monospace')
    if det_y:
        axes[1].plot(det_frames, det_y, color='#ff6600', linewidth=1, alpha=0.7, label='Detection Only')
    if kal_y:
        axes[1].plot(kal_frames, kal_y, color='#00ff00', linewidth=1.5, label='Kalman Filter')
    axes[1].set_xlabel('Frame', color='#00ff00', fontfamily='monospace')
    axes[1].set_ylabel('Y (pixels)', color='#00ff00', fontfamily='monospace')
    axes[1].legend(facecolor='#0a0a0a', edgecolor='#00ff00', labelcolor='#00ff00')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', edgecolor='none')
    plt.close()
    print(f"[INFO] 对比图表已保存: {output_path}")


# ============================================================
# 第五部分: 主程序
# Part 5: Main Application Loop
# ============================================================

def main():
    """
    主程序入口。
    初始化摄像头、人脸检测器和卡尔曼滤波器，启动实时追踪循环。
    """
    print("=" * 60)
    print("  RETRO FACE TRACKER v1.0")
    print("  基于线性代数的人脸追踪系统")
    print("=" * 60)
    print()
    print("  正在初始化系统...")

    # ----------------------------------------------------------
    # 初始化摄像头
    # ----------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头！请检查摄像头连接。")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("  [OK] 摄像头已就绪")

    # ----------------------------------------------------------
    # 加载 Haar 级联分类器（用于人脸检测）
    # OpenCV 自带的预训练模型
    # ----------------------------------------------------------
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("[ERROR] 无法加载人脸检测模型！")
        cap.release()
        return

    print("  [OK] 人脸检测模型已加载")

    # ----------------------------------------------------------
    # 初始化卡尔曼滤波器
    # ----------------------------------------------------------
    kalman = KalmanFilter2D(dt=1.0)
    print("  [OK] 卡尔曼滤波器已初始化")

    # ----------------------------------------------------------
    # 运行状态变量
    # ----------------------------------------------------------
    mode = "KALMAN"         # 当前模式: "DETECT"(纯检测) 或 "KALMAN"(卡尔曼追踪)
    svd_mode = False        # SVD 故障艺术模式开关
    svd_k = 50              # SVD 保留的奇异值数量 (默认 50)
    radar_angle = 0         # 雷达装饰动画角度

    # 用于对比实验的数据记录
    detect_positions = []   # 纯检测模式下的位置记录
    kalman_positions = []   # 卡尔曼追踪模式下的位置记录

    # FPS 计算变量
    prev_time = time.time()
    fps = 0.0

    # 上一次检测到的人脸位置（用于纯检测模式的对比记录）
    last_detect_pos = None

    print("  [OK] 系统启动完成!")
    print()
    print("  操作说明:")
    print("  [S] 切换 SVD 故障艺术模式")
    print("  [+/-] 调整 SVD 奇异值数量")
    print("  [M] 切换追踪模式")
    print("  [C] 生成对比图表")
    print("  [Q/ESC] 退出")
    print("=" * 60)

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 无法读取摄像头帧！")
            break

        # 水平翻转（镜像效果，更自然）
        frame = cv2.flip(frame, 1)

        # 计算 FPS
        current_time = time.time()
        dt = current_time - prev_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # 指数移动平均平滑 FPS
        prev_time = current_time

        # ----------------------------------------------------------
        # 人脸检测
        # 将图像转为灰度图（Haar 分类器在灰度图上工作）
        # ----------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # 图像缩放因子
            minNeighbors=5,      # 最小邻近检测次数
            minSize=(60, 60)     # 最小人脸尺寸
        )

        face_detected = len(faces) > 0

        # ----------------------------------------------------------
        # SVD 故障艺术处理（如果开启）
        # ----------------------------------------------------------
        if svd_mode:
            frame = apply_svd_compression(frame, svd_k)

        # ----------------------------------------------------------
        # 追踪逻辑
        # ----------------------------------------------------------
        if face_detected:
            # 选择最大的人脸（面积最大的检测框）
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = largest_face

            # 计算人脸中心坐标
            cx = fx + fw // 2
            cy = fy + fh // 2

            # 记录纯检测位置
            last_detect_pos = (cx, cy)
            detect_positions.append((cx, cy))

            if mode == "KALMAN":
                # ----------------------------------------------------------
                # 卡尔曼滤波模式
                # ----------------------------------------------------------
                if not kalman.initialized:
                    kalman.init_state(cx, cy)

                # 预测步骤
                kalman.predict()
                # 更新步骤（使用检测到的位置作为观测值）
                kalman.update([cx, cy])

                # 使用滤波后的位置
                kx = int(kalman.X[0, 0])
                ky = int(kalman.X[1, 0])
                kalman_positions.append((kx, ky))

                # 绘制卡尔曼追踪框
                draw_tracking_box(frame, kx - fw // 2, ky - fh // 2, fw, fh,
                                  "KALMAN LOCK", COLOR_GREEN)

                # 同时用虚线绘制原始检测位置（用于对比）
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), COLOR_DIM_GREEN, 1)

            else:
                # ----------------------------------------------------------
                # 纯检测模式
                # ----------------------------------------------------------
                draw_tracking_box(frame, fx, fy, fw, fh,
                                  "DETECT ONLY", COLOR_AMBER)

        else:
            # 没有检测到人脸
            if mode == "KALMAN" and kalman.initialized:
                # 卡尔曼滤波器继续预测（即使没有观测）
                kalman.predict()
                kx = int(kalman.X[0, 0])
                ky = int(kalman.X[1, 0])
                # 显示预测位置（虚框表示不确定）
                cv2.circle(frame, (kx, ky), 30, COLOR_RED, 1)
                cv2.putText(frame, "PREDICTING...", (kx - 50, ky - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1)

        # ----------------------------------------------------------
        # 绘制复古 UI 元素
        # ----------------------------------------------------------
        # HUD 边框
        draw_hud_border(frame)

        # 扫描线效果
        draw_scanlines(frame, intensity=20)

        # 矩阵可视化面板（右上角）
        if kalman.initialized:
            draw_matrix_display(frame, kalman, frame.shape[1] - 270, 65)

        # 雷达装饰（左下角）
        radar_angle = (radar_angle + 3) % 360
        draw_radar_decoration(frame, 70, frame.shape[0] - 110, radar_angle)

        # 底部状态栏
        draw_status_bar(frame, mode, svd_mode, svd_k, fps, face_detected)

        # ----------------------------------------------------------
        # 显示画面
        # ----------------------------------------------------------
        cv2.imshow("RETRO FACE TRACKER", frame)

        # ----------------------------------------------------------
        # 键盘事件处理
        # ----------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:  # Q 或 ESC 退出
            break

        elif key == ord('s'):  # 切换 SVD 模式
            svd_mode = not svd_mode
            status = "ON" if svd_mode else "OFF"
            print(f"[SYSTEM] SVD 故障艺术模式: {status}")

        elif key == ord('+') or key == ord('='):  # 增加 SVD 奇异值数量
            svd_k = min(svd_k + 10, 200)
            print(f"[SYSTEM] SVD 奇异值数量: k={svd_k}")

        elif key == ord('-'):  # 减少 SVD 奇异值数量
            svd_k = max(svd_k - 10, 1)
            print(f"[SYSTEM] SVD 奇异值数量: k={svd_k}")

        elif key == ord('m'):  # 切换追踪模式
            mode = "DETECT" if mode == "KALMAN" else "KALMAN"
            # 切换模式时重置卡尔曼滤波器
            if mode == "KALMAN":
                kalman = KalmanFilter2D(dt=1.0)
            print(f"[SYSTEM] 追踪模式: {mode}")

        elif key == ord('c'):  # 生成对比图表
            print("[SYSTEM] 正在生成对比图表...")
            generate_comparison_chart(detect_positions, kalman_positions)

    # ----------------------------------------------------------
    # 清理资源
    # ----------------------------------------------------------
    print()
    print("[SYSTEM] 正在关闭系统...")

    # 退出前自动生成最终对比图表
    if detect_positions or kalman_positions:
        generate_comparison_chart(detect_positions, kalman_positions)

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] 系统已安全关闭。")


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    main()
