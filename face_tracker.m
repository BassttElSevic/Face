% ================================================================================
%   ██████╗ ███████╗████████╗██████╗  ██████╗     ███████╗ █████╗  ██████╗███████╗
%   ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗    ██╔════╝██╔══██╗██╔════╝██╔════╝
%   ██████╔╝█████╗     ██║   ██████╔╝██║   ██║    █████╗  ███████║██║     █████╗
%   ██╔══██╗██╔══╝     ██║   ██╔══██╗██║   ██║    ██╔══╝  ██╔══██║██║     ██╔══╝
%   ██║  ██║███████╗   ██║   ██║  ██║╚██████╔╝    ██║     ██║  ██║╚██████╗███████╗
%   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝
%   RETRO FACE TRACKER — 基于线性代数的人脸追踪系统 (MATLAB 版本)
% ================================================================================
%
% 项目简介 / Project Overview:
%     本项目利用 MATLAB 的 Vision Toolbox 进行人脸检测，结合卡尔曼滤波器
%     (Kalman Filter) 实现平滑的人脸追踪。同时运用 SVD（奇异值分解）对
%     实时视频流进行图像压缩，展示线性代数在计算机视觉中的实际应用。
%
%     UI 采用 70-80 年代科幻复古终端风格，灵感来源于战斗机 HUD 界面。
%
% 核心线性代数知识:
%     1. 卡尔曼滤波 (Kalman Filter) — 状态预测与更新中的矩阵运算
%     2. SVD 分解 (Singular Value Decomposition) — 图像压缩与降维
%     3. 协方差矩阵 (Covariance Matrix) — 量化预测不确定性
%
% 操作说明 / Controls:
%     [S] — 切换 SVD 故障艺术模式 / Toggle SVD Glitch Art Mode
%     [+] / [-] — 增加/减少 SVD 保留的奇异值数量
%     [M] — 切换 纯检测 / 卡尔曼追踪 模式
%     [C] — 生成对比图表
%     [Q] / [ESC] — 退出程序
%
% 依赖:
%     - MATLAB R2020a 或更高版本
%     - Computer Vision Toolbox (用于人脸检测)
%     - Image Processing Toolbox
%
% 作者: 学生项目
% 日期: 2026
% ================================================================================

function face_tracker()
    %% ============================================================
    % 系统初始化
    % ============================================================
    fprintf('============================================================\n');
    fprintf('  RETRO FACE TRACKER v1.0 (MATLAB)\n');
    fprintf('  基于线性代数的人脸追踪系统\n');
    fprintf('============================================================\n\n');
    fprintf('  正在初始化系统...\n');

    % ----------------------------------------------------------
    % 初始化摄像头
    % ----------------------------------------------------------
    try
        cam = webcam(1);
        cam.Resolution = '640x480';
        fprintf('  [OK] 摄像头已就绪\n');
    catch
        error('[ERROR] 无法打开摄像头！请检查摄像头连接。');
    end

    % ----------------------------------------------------------
    % 加载人脸检测器 (使用 Viola-Jones 算法，等效于 Haar 级联)
    % ----------------------------------------------------------
    faceDetector = vision.CascadeObjectDetector();
    faceDetector.MinSize = [60 60];
    faceDetector.MergeThreshold = 5;
    faceDetector.ScaleFactor = 1.1;
    fprintf('  [OK] 人脸检测模型已加载\n');

    % ----------------------------------------------------------
    % 初始化卡尔曼滤波器
    % ----------------------------------------------------------
    kalman = init_kalman_filter(1.0);
    fprintf('  [OK] 卡尔曼滤波器已初始化\n');

    % ----------------------------------------------------------
    % 运行状态变量
    % ----------------------------------------------------------
    mode = 'KALMAN';        % 当前模式: "DETECT" 或 "KALMAN"
    svd_mode = false;       % SVD 故障艺术模式开关
    svd_k = 50;            % SVD 保留的奇异值数量
    radar_angle = 0;       % 雷达装饰动画角度

    % 用于对比实验的数据记录
    detect_positions = [];  % 纯检测模式下的位置记录 [N x 2]
    kalman_positions = [];  % 卡尔曼追踪模式下的位置记录 [N x 2]

    % FPS 计算变量
    prev_time = tic;
    fps = 0.0;

    fprintf('  [OK] 系统启动完成!\n\n');
    fprintf('  操作说明:\n');
    fprintf('  [S] 切换 SVD 故障艺术模式\n');
    fprintf('  [+/-] 调整 SVD 奇异值数量\n');
    fprintf('  [M] 切换追踪模式\n');
    fprintf('  [C] 生成对比图表\n');
    fprintf('  [Q/ESC] 退出\n');
    fprintf('============================================================\n');

    % ----------------------------------------------------------
    % 创建显示窗口
    % ----------------------------------------------------------
    hFig = figure('Name', 'RETRO FACE TRACKER', ...
                  'NumberTitle', 'off', ...
                  'MenuBar', 'none', ...
                  'Color', [0 0 0], ...
                  'KeyPressFcn', @keypress_callback, ...
                  'CloseRequestFcn', @close_callback, ...
                  'Position', [100 100 640 480]);
    hAx = axes('Parent', hFig, 'Position', [0 0 1 1]);
    axis(hAx, 'off');
    hImg = imshow(zeros(480, 640, 3, 'uint8'), 'Parent', hAx);

    % 用于键盘事件的共享变量
    running = true;
    key_pressed = '';

    %% ============================================================
    % 主循环
    % ============================================================
    while running && isvalid(hFig)
        % 读取摄像头帧
        try
            frame = snapshot(cam);
        catch
            fprintf('[ERROR] 无法读取摄像头帧！\n');
            break;
        end

        % 水平翻转（镜像效果）
        frame = fliplr(frame);

        % 计算 FPS
        dt = toc(prev_time);
        if dt > 0
            fps = 0.9 * fps + 0.1 * (1.0 / dt);
        end
        prev_time = tic;

        % ----------------------------------------------------------
        % 人脸检测
        % ----------------------------------------------------------
        gray = rgb2gray(frame);
        bboxes = step(faceDetector, gray);
        face_detected = ~isempty(bboxes);

        % ----------------------------------------------------------
        % SVD 故障艺术处理（如果开启）
        % ----------------------------------------------------------
        if svd_mode
            frame = apply_svd_compression(frame, svd_k);
        end

        % ----------------------------------------------------------
        % 追踪逻辑
        % ----------------------------------------------------------
        [h, w, ~] = size(frame);

        if face_detected
            % 选择最大的人脸
            areas = bboxes(:,3) .* bboxes(:,4);
            [~, idx] = max(areas);
            fx = bboxes(idx, 1);
            fy = bboxes(idx, 2);
            fw = bboxes(idx, 3);
            fh = bboxes(idx, 4);

            % 计算人脸中心坐标
            cx = fx + fw / 2;
            cy = fy + fh / 2;

            % 记录纯检测位置
            detect_positions = [detect_positions; cx, cy];

            if strcmp(mode, 'KALMAN')
                % 卡尔曼滤波模式
                if ~kalman.initialized
                    kalman = kalman_init_state(kalman, cx, cy);
                end

                % 预测步骤
                kalman = kalman_predict(kalman);
                % 更新步骤
                kalman = kalman_update(kalman, [cx; cy]);

                % 使用滤波后的位置
                kx = kalman.X(1);
                ky = kalman.X(2);
                kalman_positions = [kalman_positions; kx, ky];

                % 绘制卡尔曼追踪框
                frame = draw_tracking_box(frame, round(kx - fw/2), round(ky - fh/2), ...
                                          fw, fh, 'KALMAN LOCK', [0 255 0]);

                % 绘制原始检测位置（用于对比）
                frame = draw_rectangle(frame, fx, fy, fw, fh, [0 100 0], 1);

            else
                % 纯检测模式
                frame = draw_tracking_box(frame, fx, fy, fw, fh, ...
                                          'DETECT ONLY', [0 200 255]);
            end

        else
            % 没有检测到人脸
            if strcmp(mode, 'KALMAN') && kalman.initialized
                kalman = kalman_predict(kalman);
                kx = round(kalman.X(1));
                ky = round(kalman.X(2));
                % 显示预测位置
                frame = insertShape(frame, 'Circle', [kx ky 30], ...
                                    'Color', [255 0 0], 'LineWidth', 1);
                frame = insertText(frame, [kx-50 ky-35], 'PREDICTING...', ...
                                   'FontSize', 10, 'TextColor', [255 0 0], ...
                                   'BoxOpacity', 0);
            end
        end

        % ----------------------------------------------------------
        % 绘制复古 UI 元素
        % ----------------------------------------------------------
        frame = draw_hud_border(frame, w, h);
        frame = draw_scanlines(frame, 20);

        % 矩阵可视化面板（右上角）
        if kalman.initialized
            frame = draw_matrix_display(frame, kalman, w - 270, 65);
        end

        % 雷达装饰（左下角）
        radar_angle = mod(radar_angle + 3, 360);
        frame = draw_radar_decoration(frame, 70, h - 110, radar_angle);

        % 底部状态栏
        frame = draw_status_bar(frame, mode, svd_mode, svd_k, fps, face_detected, w, h);

        % ----------------------------------------------------------
        % 显示画面
        % ----------------------------------------------------------
        set(hImg, 'CData', frame);
        drawnow limitrate;

        % ----------------------------------------------------------
        % 处理键盘事件
        % ----------------------------------------------------------
        if ~isempty(key_pressed)
            switch key_pressed
                case 'q'
                    running = false;
                case 'escape'
                    running = false;
                case 's'
                    svd_mode = ~svd_mode;
                    if svd_mode
                        fprintf('[SYSTEM] SVD 故障艺术模式: ON\n');
                    else
                        fprintf('[SYSTEM] SVD 故障艺术模式: OFF\n');
                    end
                case 'equal' % +
                    svd_k = min(svd_k + 10, 200);
                    fprintf('[SYSTEM] SVD 奇异值数量: k=%d\n', svd_k);
                case 'hyphen' % -
                    svd_k = max(svd_k - 10, 1);
                    fprintf('[SYSTEM] SVD 奇异值数量: k=%d\n', svd_k);
                case 'add' % numpad +
                    svd_k = min(svd_k + 10, 200);
                    fprintf('[SYSTEM] SVD 奇异值数量: k=%d\n', svd_k);
                case 'subtract' % numpad -
                    svd_k = max(svd_k - 10, 1);
                    fprintf('[SYSTEM] SVD 奇异值数量: k=%d\n', svd_k);
                case 'm'
                    if strcmp(mode, 'KALMAN')
                        mode = 'DETECT';
                    else
                        mode = 'KALMAN';
                        kalman = init_kalman_filter(1.0);
                    end
                    fprintf('[SYSTEM] 追踪模式: %s\n', mode);
                case 'c'
                    fprintf('[SYSTEM] 正在生成对比图表...\n');
                    generate_comparison_chart(detect_positions, kalman_positions);
            end
            key_pressed = '';
        end
    end

    %% ============================================================
    % 清理资源
    % ============================================================
    fprintf('\n[SYSTEM] 正在关闭系统...\n');

    % 退出前自动生成最终对比图表
    if ~isempty(detect_positions) || ~isempty(kalman_positions)
        generate_comparison_chart(detect_positions, kalman_positions);
    end

    clear cam;
    if isvalid(hFig)
        delete(hFig);
    end
    fprintf('[SYSTEM] 系统已安全关闭。\n');

    %% ============================================================
    % 嵌套回调函数
    % ============================================================
    function keypress_callback(~, event)
        key_pressed = event.Key;
    end

    function close_callback(~, ~)
        running = false;
    end
end


%% ============================================================
% 第一部分: 卡尔曼滤波器函数
% Part 1: Kalman Filter Functions
% ============================================================
% 卡尔曼滤波器是一种最优线性状态估计器，核心是通过"预测-更新"
% 两个步骤，利用矩阵运算不断修正对目标状态的估计。
% ============================================================

function kalman = init_kalman_filter(dt)
    % 初始化卡尔曼滤波器
    %
    % 参数:
    %     dt: 时间步长
    %
    % 返回:
    %     kalman: 包含所有滤波器参数的结构体

    % ----------------------------------------------------------
    % 状态向量 X: 4×1 矩阵 (列向量)
    % X = [x, y, vx, vy]^T
    % ----------------------------------------------------------
    kalman.X = zeros(4, 1);

    % ----------------------------------------------------------
    % 状态转移矩阵 F: 4×4 矩阵
    % 描述匀速运动模型: x_new = x + vx * dt
    %                    y_new = y + vy * dt
    % ----------------------------------------------------------
    kalman.F = [1, 0, dt, 0;
                0, 1, 0,  dt;
                0, 0, 1,  0;
                0, 0, 0,  1];

    % ----------------------------------------------------------
    % 观测矩阵 H: 2×4 矩阵
    % 只能观测位置 (x, y)，不能直接观测速度
    % ----------------------------------------------------------
    kalman.H = [1, 0, 0, 0;
                0, 1, 0, 0];

    % ----------------------------------------------------------
    % 协方差矩阵 P: 4×4 对称正定矩阵
    % 初始设为较大值，表示初始状态非常不确定
    % ----------------------------------------------------------
    kalman.P = eye(4) * 1000.0;

    % ----------------------------------------------------------
    % 过程噪声协方差矩阵 Q: 4×4 矩阵
    % ----------------------------------------------------------
    kalman.Q = eye(4) * 0.1;
    kalman.Q(1,1) = 0.5;   % 位置 x 的噪声
    kalman.Q(2,2) = 0.5;   % 位置 y 的噪声
    kalman.Q(3,3) = 1.0;   % 速度 vx 的噪声
    kalman.Q(4,4) = 1.0;   % 速度 vy 的噪声

    % ----------------------------------------------------------
    % 观测噪声协方差矩阵 R: 2×2 矩阵
    % ----------------------------------------------------------
    kalman.R = eye(2) * 5.0;

    % ----------------------------------------------------------
    % 初始化标志
    % ----------------------------------------------------------
    kalman.initialized = false;
end


function kalman = kalman_predict(kalman)
    % 预测步骤 — 卡尔曼滤波的第一阶段
    %
    % 数学公式:
    %     X_pred = F * X          (状态预测: 矩阵-向量乘法)
    %     P_pred = F * P * F' + Q (协方差预测: 矩阵乘法与转置)
    %
    % 线性代数知识:
    %     - F * X 是矩阵与向量的乘法，实现线性变换
    %     - F * P * F' 是二次型变换，保持 P 的对称正定性
    %     - F' 是 F 的转置矩阵

    % 状态预测: X_pred = F * X
    kalman.X = kalman.F * kalman.X;

    % 协方差预测: P_pred = F * P * F' + Q
    kalman.P = kalman.F * kalman.P * kalman.F' + kalman.Q;
end


function kalman = kalman_update(kalman, z)
    % 更新步骤 — 卡尔曼滤波的第二阶段
    %
    % 参数:
    %     z: 观测向量 [x_obs; y_obs]，即检测到的人脸中心坐标
    %
    % 数学公式:
    %     y = z - H * X           (新息/残差)
    %     S = H * P * H' + R     (新息协方差矩阵)
    %     K = P * H' * S^(-1)    (卡尔曼增益: 需要矩阵求逆)
    %     X = X + K * y           (状态更新)
    %     P = (I - K * H) * P     (协方差更新)

    z = z(:);  % 确保为列向量

    % 计算新息（残差）: y = z - H * X
    y = z - kalman.H * kalman.X;

    % 计算新息协方差: S = H * P * H' + R
    S = kalman.H * kalman.P * kalman.H' + kalman.R;

    % 计算卡尔曼增益: K = P * H' * S^(-1)
    K = (kalman.P * kalman.H') / S;

    % 状态更新: X = X + K * y
    kalman.X = kalman.X + K * y;

    % 协方差更新: P = (I - K * H) * P
    I4 = eye(4);
    kalman.P = (I4 - K * kalman.H) * kalman.P;
end


function kalman = kalman_init_state(kalman, x, y)
    % 使用第一次检测到的人脸位置初始化状态向量
    kalman.X = [x; y; 0; 0];
    kalman.initialized = true;
end


%% ============================================================
% 第二部分: SVD 图像压缩
% Part 2: SVD Image Compression (Glitch Art)
% ============================================================
% SVD（奇异值分解）将任意 m×n 矩阵 A 分解为:
%     A = U * Σ * V'
% 通过只保留前 k 个最大的奇异值，可以实现图像的低秩近似
% ============================================================

function compressed = apply_svd_compression(frame, k)
    % 对图像帧应用 SVD 压缩（故障艺术效果）
    %
    % 参数:
    %     frame: 输入图像 (RGB uint8)
    %     k: 保留的奇异值数量
    %
    % 返回:
    %     compressed: 压缩后的图像

    compressed = zeros(size(frame), 'uint8');

    % 对 R, G, B 三个通道分别进行 SVD 分解
    for i = 1:3
        % 提取单个颜色通道（m×n 实数矩阵）
        channel = double(frame(:,:,i));

        % ----------------------------------------------------------
        % 执行 SVD 分解: Channel = U * S * V'
        % U: 左奇异向量矩阵
        % S: 奇异值对角矩阵
        % V: 右奇异向量矩阵
        % 使用 'econ' 选项返回经济型 SVD，节省内存
        % ----------------------------------------------------------
        [U, S, V] = svd(channel, 'econ');

        % 限制 k 不超过奇异值的总数
        k_actual = min(k, size(S, 1));

        % ----------------------------------------------------------
        % 低秩近似: 只保留前 k 个奇异值
        % Channel_approx = U(:,1:k) * S(1:k,1:k) * V(:,1:k)'
        % ----------------------------------------------------------
        channel_compressed = U(:,1:k_actual) * S(1:k_actual,1:k_actual) * V(:,1:k_actual)';

        % 将像素值裁剪到 [0, 255] 范围并转为 uint8
        compressed(:,:,i) = uint8(max(0, min(255, channel_compressed)));
    end
end


%% ============================================================
% 第三部分: 复古终端 UI 绘制
% Part 3: Retro Terminal UI Rendering
% ============================================================

function frame = draw_scanlines(frame, intensity)
    % 绘制 CRT 扫描线效果
    [h, w, ~] = size(frame);
    alpha = intensity / 255.0;

    % 每隔 2 行变暗，模拟 CRT 扫描线
    frame(1:2:h, :, :) = uint8(double(frame(1:2:h, :, :)) * (1 - alpha));
end


function frame = draw_hud_border(frame, w, h)
    % 绘制 HUD 风格的边框
    COLOR_GREEN = [0 255 0];
    COLOR_DIM_GREEN = [0 100 0];
    COLOR_AMBER = [255 200 0];  % RGB 格式

    % 外层边框
    frame = draw_rect_outline(frame, 5, 5, w-10, h-10, COLOR_DIM_GREEN, 1);
    % 内层边框
    frame = draw_rect_outline(frame, 15, 15, w-30, h-30, COLOR_GREEN, 1);

    corner_len = 30;

    % 四个角落的 L 形装饰
    % 左上角
    frame = draw_line(frame, 15, 15, 15+corner_len, 15, COLOR_AMBER, 2);
    frame = draw_line(frame, 15, 15, 15, 15+corner_len, COLOR_AMBER, 2);
    % 右上角
    frame = draw_line(frame, w-15, 15, w-15-corner_len, 15, COLOR_AMBER, 2);
    frame = draw_line(frame, w-15, 15, w-15, 15+corner_len, COLOR_AMBER, 2);
    % 左下角
    frame = draw_line(frame, 15, h-15, 15+corner_len, h-15, COLOR_AMBER, 2);
    frame = draw_line(frame, 15, h-15, 15, h-15-corner_len, COLOR_AMBER, 2);
    % 右下角
    frame = draw_line(frame, w-15, h-15, w-15-corner_len, h-15, COLOR_AMBER, 2);
    frame = draw_line(frame, w-15, h-15, w-15, h-15-corner_len, COLOR_AMBER, 2);

    % 顶部标题栏
    frame = draw_filled_rect(frame, 20, 20, w-40, 35, [0 0 0]);
    frame = draw_rect_outline(frame, 20, 20, w-40, 35, COLOR_GREEN, 1);
    frame = insertText(frame, [30 25], '[ RETRO FACE TRACKER v1.0 ]', ...
                       'FontSize', 14, 'TextColor', COLOR_GREEN, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % 显示当前时间戳
    timestamp = char(datetime('now', 'Format', 'HH:mm:ss'));
    frame = insertText(frame, [w-230 25], ['SYS TIME: ' timestamp], ...
                       'FontSize', 12, 'TextColor', COLOR_AMBER, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');
end


function frame = draw_crosshair(frame, cx, cy, sz, color)
    % 在目标位置绘制十字准星（战斗机 HUD 风格）
    half = round(sz / 2);
    gap = 8;

    % 带间隙的十字线
    frame = draw_line(frame, cx-half, cy, cx-gap, cy, color, 1);
    frame = draw_line(frame, cx+gap, cy, cx+half, cy, color, 1);
    frame = draw_line(frame, cx, cy-half, cx, cy-gap, color, 1);
    frame = draw_line(frame, cx, cy+gap, cx, cy+half, color, 1);

    % 角落小方块
    bracket = 10;
    frame = draw_line(frame, cx-half, cy-half, cx-half+bracket, cy-half, color, 1);
    frame = draw_line(frame, cx-half, cy-half, cx-half, cy-half+bracket, color, 1);
    frame = draw_line(frame, cx+half, cy-half, cx+half-bracket, cy-half, color, 1);
    frame = draw_line(frame, cx+half, cy-half, cx+half, cy-half+bracket, color, 1);
    frame = draw_line(frame, cx-half, cy+half, cx-half+bracket, cy+half, color, 1);
    frame = draw_line(frame, cx-half, cy+half, cx-half, cy+half-bracket, color, 1);
    frame = draw_line(frame, cx+half, cy+half, cx+half-bracket, cy+half, color, 1);
    frame = draw_line(frame, cx+half, cy+half, cx+half, cy+half-bracket, color, 1);
end


function frame = draw_tracking_box(frame, x, y, w, h, label, color)
    % 绘制复古风格的追踪框
    corner = 15;

    % 四个角的 L 形线段
    % 左上
    frame = draw_line(frame, x, y, x+corner, y, color, 2);
    frame = draw_line(frame, x, y, x, y+corner, color, 2);
    % 右上
    frame = draw_line(frame, x+w, y, x+w-corner, y, color, 2);
    frame = draw_line(frame, x+w, y, x+w, y+corner, color, 2);
    % 左下
    frame = draw_line(frame, x, y+h, x+corner, y+h, color, 2);
    frame = draw_line(frame, x, y+h, x, y+h-corner, color, 2);
    % 右下
    frame = draw_line(frame, x+w, y+h, x+w-corner, y+h, color, 2);
    frame = draw_line(frame, x+w, y+h, x+w, y+h-corner, color, 2);

    % 标签
    frame = insertText(frame, [x, y-22], label, ...
                       'FontSize', 10, 'TextColor', color, ...
                       'BoxColor', [0 0 0], 'BoxOpacity', 0.8, ...
                       'Font', 'Courier New');

    % 中心十字准星
    cx = round(x + w/2);
    cy = round(y + h/2);
    frame = draw_crosshair(frame, cx, cy, round(min(w,h)/2), color);
end


function frame = draw_matrix_display(frame, kalman, x_offset, y_offset)
    % 实时显示卡尔曼滤波器的状态向量 X 和协方差矩阵 P
    COLOR_GREEN = [0 255 0];
    COLOR_AMBER = [255 200 0];
    COLOR_RED = [255 0 0];

    panel_w = 250;
    panel_h = 180;

    % 半透明背景面板
    frame = draw_filled_rect_alpha(frame, x_offset, y_offset, panel_w, panel_h, [0 0 0], 0.7);

    % 面板边框
    frame = draw_rect_outline(frame, x_offset, y_offset, panel_w, panel_h, COLOR_GREEN, 1);

    % 标题
    frame = insertText(frame, [x_offset+10, y_offset+5], '[ STATE VECTOR X ]', ...
                       'FontSize', 10, 'TextColor', COLOR_AMBER, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % 显示状态向量的每个分量
    labels = {'POS_X', 'POS_Y', 'VEL_X', 'VEL_Y'};
    for i = 1:4
        val = kalman.X(i);
        y_pos = y_offset + 25 + i * 22;
        txt = sprintf('%s: %8.2f', labels{i}, val);
        frame = insertText(frame, [x_offset+15, y_pos], txt, ...
                           'FontSize', 9, 'TextColor', COLOR_GREEN, ...
                           'BoxOpacity', 0, 'Font', 'Courier New');

        % 小型条形图
        bar_len = round(min(abs(val)/5.0, 1.0) * 80);
        if val >= 0
            bar_color = COLOR_GREEN;
        else
            bar_color = COLOR_RED;
        end
        if bar_len > 0
            frame = draw_filled_rect(frame, x_offset+155, y_pos+2, bar_len, 8, bar_color);
        end
    end

    % 协方差矩阵 P 热力图
    frame = insertText(frame, [x_offset+10, y_offset+130], '[ COVARIANCE P ]', ...
                       'FontSize', 10, 'TextColor', COLOR_AMBER, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    diag_vals = diag(kalman.P);
    max_val = max(max(abs(diag_vals)), 1.0);
    for i = 1:4
        intensity_val = min(round(abs(diag_vals(i)) / max_val * 255), 255);
        color_block = [0 intensity_val 0];
        bx = x_offset + 15 + (i-1) * 55;
        by = y_offset + 150;
        frame = draw_filled_rect(frame, bx, by, 45, 20, color_block);
        frame = draw_rect_outline(frame, bx, by, 45, 20, COLOR_GREEN, 1);
        frame = insertText(frame, [bx+2, by+2], sprintf('%.1f', diag_vals(i)), ...
                           'FontSize', 8, 'TextColor', COLOR_AMBER, ...
                           'BoxOpacity', 0, 'Font', 'Courier New');
    end
end


function frame = draw_status_bar(frame, mode, svd_mode, svd_k, fps, face_detected, w, h)
    % 绘制底部状态栏
    COLOR_GREEN = [0 255 0];
    COLOR_AMBER = [255 200 0];
    COLOR_CYAN = [0 255 255];
    COLOR_DIM_GREEN = [0 100 0];
    COLOR_RED = [255 0 0];

    bar_y = h - 65;

    % 底部状态栏背景
    frame = draw_filled_rect_alpha(frame, 15, bar_y, w-30, 50, [0 0 0], 0.75);
    frame = draw_rect_outline(frame, 15, bar_y, w-30, 50, COLOR_GREEN, 1);

    % 模式指示
    if strcmp(mode, 'KALMAN')
        mode_color = COLOR_GREEN;
    else
        mode_color = COLOR_AMBER;
    end
    frame = insertText(frame, [25, bar_y+5], ['MODE: ' mode], ...
                       'FontSize', 10, 'TextColor', mode_color, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % SVD 状态
    if svd_mode
        svd_text = sprintf('SVD: ON (k=%d)', svd_k);
        svd_color = COLOR_CYAN;
    else
        svd_text = 'SVD: OFF';
        svd_color = COLOR_DIM_GREEN;
    end
    frame = insertText(frame, [180, bar_y+5], svd_text, ...
                       'FontSize', 10, 'TextColor', svd_color, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % FPS 显示
    frame = insertText(frame, [380, bar_y+5], sprintf('FPS: %.1f', fps), ...
                       'FontSize', 10, 'TextColor', COLOR_GREEN, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % 人脸检测状态
    if face_detected
        status = 'LOCKED';
        status_color = COLOR_GREEN;
    else
        status = 'SCANNING...';
        status_color = COLOR_RED;
    end
    frame = insertText(frame, [480, bar_y+5], ['TARGET: ' status], ...
                       'FontSize', 10, 'TextColor', status_color, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');

    % 控制提示
    frame = insertText(frame, [25, bar_y+28], '[S]SVD  [+/-]K  [M]MODE  [C]CHART  [Q]QUIT', ...
                       'FontSize', 9, 'TextColor', COLOR_DIM_GREEN, ...
                       'BoxOpacity', 0, 'Font', 'Courier New');
end


function frame = draw_radar_decoration(frame, cx, cy, angle)
    % 绘制雷达扫描动画装饰
    COLOR_DIM_GREEN = [0 100 0];
    COLOR_GREEN = [0 255 0];
    radius = 40;

    % 同心圆
    frame = insertShape(frame, 'Circle', [cx cy radius], ...
                        'Color', COLOR_DIM_GREEN, 'LineWidth', 1);
    frame = insertShape(frame, 'Circle', [cx cy round(radius/2)], ...
                        'Color', COLOR_DIM_GREEN, 'LineWidth', 1);

    % 十字线
    frame = draw_line(frame, cx-radius, cy, cx+radius, cy, COLOR_DIM_GREEN, 1);
    frame = draw_line(frame, cx, cy-radius, cx, cy+radius, COLOR_DIM_GREEN, 1);

    % 扫描线（旋转的半径线）
    end_x = round(cx + radius * cosd(angle));
    end_y = round(cy + radius * sind(angle));
    frame = draw_line(frame, cx, cy, end_x, end_y, COLOR_GREEN, 1);
end


%% ============================================================
% 第四部分: 对比实验与图表生成
% Part 4: Comparison Experiment & Chart Generation
% ============================================================

function generate_comparison_chart(detect_positions, kalman_positions)
    % 生成对比图表，展示纯检测 vs 卡尔曼滤波追踪的位置变化曲线
    %
    % 参数:
    %     detect_positions: 纯检测模式下的位置记录 [N x 2]
    %     kalman_positions: 卡尔曼滤波模式下的位置记录 [N x 2]

    fig = figure('Visible', 'off', 'Position', [0 0 1200 800]);
    set(fig, 'Color', [0.04 0.04 0.04]);

    % X 坐标对比图
    ax1 = subplot(2, 1, 1);
    set(ax1, 'Color', [0.04 0.04 0.04]);
    hold(ax1, 'on');
    grid(ax1, 'on');
    set(ax1, 'GridColor', [0 0.2 0], 'GridAlpha', 0.5, 'GridLineStyle', '--');
    set(ax1, 'XColor', [0 1 0], 'YColor', [0 1 0]);

    if ~isempty(detect_positions)
        plot(ax1, 1:size(detect_positions,1), detect_positions(:,1), ...
             'Color', [1 0.4 0], 'LineWidth', 1);
    end
    if ~isempty(kalman_positions)
        plot(ax1, 1:size(kalman_positions,1), kalman_positions(:,1), ...
             'Color', [0 1 0], 'LineWidth', 1.5);
    end
    title(ax1, 'X Position Over Time', 'Color', [0 1 0], 'FontSize', 14, 'FontName', 'Courier New');
    ylabel(ax1, 'X (pixels)', 'Color', [0 1 0], 'FontName', 'Courier New');
    legend(ax1, {'Detection Only', 'Kalman Filter'}, ...
           'TextColor', [0 1 0], 'Color', [0.04 0.04 0.04], ...
           'EdgeColor', [0 1 0]);
    hold(ax1, 'off');

    % Y 坐标对比图
    ax2 = subplot(2, 1, 2);
    set(ax2, 'Color', [0.04 0.04 0.04]);
    hold(ax2, 'on');
    grid(ax2, 'on');
    set(ax2, 'GridColor', [0 0.2 0], 'GridAlpha', 0.5, 'GridLineStyle', '--');
    set(ax2, 'XColor', [0 1 0], 'YColor', [0 1 0]);

    if ~isempty(detect_positions)
        plot(ax2, 1:size(detect_positions,1), detect_positions(:,2), ...
             'Color', [1 0.4 0], 'LineWidth', 1);
    end
    if ~isempty(kalman_positions)
        plot(ax2, 1:size(kalman_positions,1), kalman_positions(:,2), ...
             'Color', [0 1 0], 'LineWidth', 1.5);
    end
    title(ax2, 'Y Position Over Time', 'Color', [0 1 0], 'FontSize', 14, 'FontName', 'Courier New');
    xlabel(ax2, 'Frame', 'Color', [0 1 0], 'FontName', 'Courier New');
    ylabel(ax2, 'Y (pixels)', 'Color', [0 1 0], 'FontName', 'Courier New');
    legend(ax2, {'Detection Only', 'Kalman Filter'}, ...
           'TextColor', [0 1 0], 'Color', [0.04 0.04 0.04], ...
           'EdgeColor', [0 1 0]);
    hold(ax2, 'off');

    % 保存图表
    output_path = 'comparison_chart.png';
    print(fig, output_path, '-dpng', '-r150');
    close(fig);
    fprintf('[INFO] 对比图表已保存: %s\n', output_path);
end


%% ============================================================
% 辅助绘图函数
% Helper Drawing Functions
% ============================================================

function frame = draw_line(frame, x1, y1, x2, y2, color, thickness)
    % 在图像上绘制线段 (使用 insertShape)
    [h, w, ~] = size(frame);

    % 裁剪坐标到图像范围
    x1 = max(1, min(w, round(x1)));
    y1 = max(1, min(h, round(y1)));
    x2 = max(1, min(w, round(x2)));
    y2 = max(1, min(h, round(y2)));

    frame = insertShape(frame, 'Line', [x1 y1 x2 y2], ...
                        'Color', color, 'LineWidth', thickness);
end


function frame = draw_rectangle(frame, x, y, w, h, color, thickness)
    % 绘制矩形边框
    [fh, fw, ~] = size(frame);
    x = max(1, min(fw, round(x)));
    y = max(1, min(fh, round(y)));
    w = max(1, round(w));
    h = max(1, round(h));

    frame = insertShape(frame, 'Rectangle', [x y w h], ...
                        'Color', color, 'LineWidth', thickness);
end


function frame = draw_rect_outline(frame, x, y, w, h, color, thickness)
    % 绘制矩形边框
    frame = draw_rectangle(frame, x, y, w, h, color, thickness);
end


function frame = draw_filled_rect(frame, x, y, w, h, color)
    % 绘制填充矩形
    [fh, fw, ~] = size(frame);

    % 裁剪坐标
    x1 = max(1, round(x));
    y1 = max(1, round(y));
    x2 = min(fw, round(x + w));
    y2 = min(fh, round(y + h));

    if x1 >= x2 || y1 >= y2
        return;
    end

    frame(y1:y2, x1:x2, 1) = color(1);
    frame(y1:y2, x1:x2, 2) = color(2);
    frame(y1:y2, x1:x2, 3) = color(3);
end


function frame = draw_filled_rect_alpha(frame, x, y, w, h, color, alpha)
    % 绘制半透明填充矩形
    [fh, fw, ~] = size(frame);

    % 裁剪坐标
    x1 = max(1, round(x));
    y1 = max(1, round(y));
    x2 = min(fw, round(x + w));
    y2 = min(fh, round(y + h));

    if x1 >= x2 || y1 >= y2
        return;
    end

    % 混合原图和填充色
    for c = 1:3
        frame(y1:y2, x1:x2, c) = uint8(alpha * double(color(c)) + ...
                                        (1 - alpha) * double(frame(y1:y2, x1:x2, c)));
    end
end
