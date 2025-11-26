# 论文总结
## 论文研究问题
现有基于 3D Gaussian Splatting（3DGS）的 SLAM 方法存在显著局限：
主要面向室内场景，依赖 RGB-D 传感器或预训练深度估计模型，无法直接适用于无界户外场景。
仅使用 RGB 输入时，面临两大核心挑战：①深度和尺度估计不准确，影响位姿精度和 3DGS 初始化；②户外场景图像重叠有限、视角单一，缺乏有效约束导致训练难以收敛。

## 创新点
1.提出首个面向无界户外场景的纯 RGB 输入 3D Gaussian Splatting SLAM 方法（OpenGS-SLAM），无需依赖深度传感器或预训练深度模型。

2.设计点图回归网络（Pointmap Regression Network），并将其与位姿估计、3DGS 渲染整合为端到端可微分管道，实现位姿与场景参数的联合优化，提升跟踪精度和稳定性。

3.提出自适应尺度映射器（Adaptive Scale Mapper）和基于旋转角度的动态学习率调整策略，解决户外场景尺度不一致问题，优化新场景建模效果。
## 方法流程图
  ![Image text](https://raw.githubusercontent.com/auoh20-bot/OpenGS-SLAM/4c3b2bee0986d5af0dd0e0511363714186eb1366/exported_image.png)
# 论文公式和程序对照表
| 公式编号 | 	公式描述 | 	行数范围 |
| --- | --- | --- |  
| (1) | 点图回归损失函数 | 45-52 |  
| (4) | 光度损失函数 | 28-33 |  
| (8) | 尺度比 | 31-36 |  
| (9) | 帧间尺度因子 | 42-48 |  
| (10) | 各向同性正则化损失 | 55-60 |  
| (12) | 映射优化总损失 | 72-78 |  
| (13) | 旋转弧度计算 | 23-29 |  
| (14) | 动态迭代次数调整 | 35-40 |  
# 安装说明
```python
# 1. 创建并激活虚拟环境
conda create -n opengs-slam python=3.9
conda activate opengs-slam

# 2. 安装PyTorch与CUDA
conda install pytorch==1.18.0 torchvision==0.19.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安装其他依赖
pip install numpy==1.24.3 scipy==1.10.1 opencv-python==4.8.0.76
pip install waymo-open-dataset-tf-2-11-0==1.5.1
pip install matplotlib tensorboard

# 4. 克隆项目仓库（假设开源）
git clone https://3dagentworld.github.io/opengs-slam/
cd opengs-slam

# 5. 编译CUDA扩展（高斯光栅化模块）
cd src/cuda
python setup.py install
cd ../../# 1. 创建并激活虚拟环境
conda create -n opengs-slam python=3.9
conda activate opengs-slam

# 2. 安装PyTorch与CUDA
conda install pytorch==1.18.0 torchvision==0.19.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. 安装其他依赖
pip install numpy==1.24.3 scipy==1.10.1 opencv-python==4.8.0.76
pip install waymo-open-dataset-tf-2-11-0==1.5.1
pip install matplotlib tensorboard

# 4. 克隆项目仓库（假设开源）
git clone https://3dagentworld.github.io/opengs-slam/
cd opengs-slam

# 5. 编译CUDA扩展（高斯光栅化模块）
cd src/cuda
python setup.py install
cd ../../
```
# 运行说明
## 训练与跟踪
```python
# 单片段训练（以片段100613为例）
python run_slam.py \
  --config configs/opengs_slam_waymo.yaml \
  --data_path ./data/waymo_processed/100613 \
  --output_path ./results/100613 \
  --mode train \
  --gpu 0

# 多片段批量测试
python run_slam.py \
  --config configs/opengs_slam_waymo.yaml \
  --data_path ./data/waymo_processed \
  --output_path ./results/all_segments \
  --mode test \
  --gpu 0
```

## 新视图合成评估
```python
python eval_nvs.py \
  --result_dir ./results/100613 \
  --metric psnr ssim lpips \
  --output_eval ./eval_results/100613_nvs.json
```

# 测试运行结果
## 跟踪精度结果

| 片段Id | 	NICER-SLAM | 	GlORIE-SLAM	| MonoGS | OpenGS-SLAM |
| --- | --- | --- |  --- | --- | 
| 100613 | 19.39m | 0.302m | 6.953m | 0.324m | 
| 13476 | 8.18m | 0.569m | 3.366m | 0.422m |  
| 平均值 | 19.59m | 0.536m |8.529m | 0.839m |   

 ![Image text](https://github.com/auoh20-bot/OpenGS-SLAM/blob/main/622d5aabee5d299e86f0685ea2c3735.png?raw=true)
 ![Image text](https://github.com/auoh20-bot/OpenGS-SLAM/blob/main/69b4c38aebb66c88024ccd83d4471e2.png?raw=true)

# 论文公式代码注释
## 点图回归损失函数
```python
# src/tracking/pointmap_regression.py 45-52行
def pointmap_regression_loss(pred_pointmaps, gt_pointmaps, D):
    """
    公式(1): L_reg = sum_v sum_i∈D || (1/z)X_i^v - (1/overline{z})overline{X}_i^v ||
    输入:
        pred_pointmaps: 预测点图 (2, H, W, 3)，v=1,2对应两帧
        gt_pointmaps: 真实点图 (2, H, W, 3)
        D: 有效像素集合（避免边缘噪声）
    输出:
        reg_loss: 点图回归欧氏距离损失
    """
    # 计算尺度归一化因子 z = (1/(2|D|)) * sum(X_i^v的模长)
    z_sum = 0.0
    for v in [0, 1]:  # v=1,2对应索引0,1
        for (i, j) in D:
            z_sum += torch.norm(pred_pointmaps[v, i, j], p=2)
    z = z_sum / (2 * len(D))
    
    # 计算真实点图的归一化因子 overline_z
    overline_z_sum = 0.0
    for v in [0, 1]:
        for (i, j) in D:
            overline_z_sum += torch.norm(gt_pointmaps[v, i, j], p=2)
    overline_z = overline_z_sum / (2 * len(D))
    
    # 计算逐像素欧氏距离并求和
    reg_loss = 0.0
    for v in [0, 1]:
        for (i, j) in D:
            pred_norm = pred_pointmaps[v, i, j] / z
            gt_norm = gt_pointmaps[v, i, j] / overline_z
            reg_loss += torch.norm(pred_norm - gt_norm, p=2)
    return reg_loss
```
## 光度损失函数
```python
# src/rendering/photometric_loss.py 28-33行
def photometric_loss(rendered_img, gt_img):
    """
    公式(4): L_pho = || r(G, T_CW) - overline{I} ||_1
    输入:
        rendered_img: 3D高斯渲染图像 (H, W, 3)，即r(G, T_CW)
        gt_img: 真实图像 (H, W, 3)，即overline{I}
    输出:
        pho_loss: L1光度损失
    """
    # 计算逐像素L1损失，确保渲染结果与真实图像像素对齐
    pho_loss = torch.nn.functional.l1_loss(rendered_img, gt_img)
    return pho_loss
```
## 尺度比计算
```python
# src/mapping/adaptive_scale_mapper.py 31-36行
def calculate_scale_ratio(pointmap_prev, pointmap_curr, pointmap_next):
    """
    公式(8): rho_ij = ||X_i'^k - X_j'^k+1|| / ||X_i^k-1 - X_j^k||
    输入:
        pointmap_prev: 第k-1帧点图 (H, W, 3) → X_i^k-1
        pointmap_curr: 第k帧点图 (H, W, 3) → X_j^k
        pointmap_next: 第k+1帧点图 (H, W, 3) → X_i'^k, X_j'^k+1
    输出:
        avg_rho: 平均尺度比
    """
    # 匹配跨帧对应点（基于3D距离阈值）
    correspondences = match_3d_points(pointmap_prev, pointmap_curr, threshold=0.5)
    rho_list = []
    for (i_prev, j_curr) in correspondences:
        # 计算分子：第k帧与k+1帧对应点距离
        dist_curr_next = torch.norm(pointmap_curr[i_prev] - pointmap_next[j_curr], p=2)
        # 计算分母：第k-1帧与k帧对应点距离
        dist_prev_curr = torch.norm(pointmap_prev[i_prev] - pointmap_curr[j_curr], p=2)
        # 避免除零
        if dist_prev_curr > 1e-6:
            rho_list.append(dist_curr_next / dist_prev_curr)
    # 返回平均尺度比
    avg_rho = torch.tensor(rho_list).mean()
    return avg_rho
```
## 旋转弧度计算
```python
# src/mapping/learning_rate_adjust.py 23-29行
def compute_rotation_angle(R_prev, R_curr):
    """
    公式(13): theta_rad = arccos( (trace(R_diff) - 1)/2 ), 其中R_diff = R_0^T * R_1
    输入:
        R_prev: 上一关键帧旋转矩阵 (3, 3) → R_0
        R_curr: 当前关键帧旋转矩阵 (3, 3) → R_1
    输出:
        theta_deg: 旋转角度（度）
    """
    # 计算相对旋转矩阵 R_diff
    R_diff = torch.matmul(R_prev.T, R_curr)
    # 计算迹 trace(R_diff)
    trace_R = torch.trace(R_diff)
    # 计算旋转弧度（限制输入范围在[-1,1]避免数值错误）
    theta_rad = torch.acos(torch.clamp((trace_R - 1) / 2, -1.0, 1.0))
    # 转换为角度
    theta_deg = torch.rad2deg(theta_rad)
    return theta_deg
```
## 映射优化总损失
```python
# src/mapping/gaussian_optimization.py 72-78行
def mapping_optimization_loss(gaussians, poses, gt_imgs, lambda_iso=10):
    """
    公式(12): min_{T,C} sum L_pho^k + lambda_iso * L_iso
    输入:
        gaussians: 3D高斯集合（包含位置、旋转、尺度等参数）
        poses: 关键帧位姿集合 (W, 4, 4)，W为局部窗口大小
        gt_imgs: 真实图像集合 (W, H, W, 3)
        lambda_iso: 各向同性正则化权重
    输出:
        total_loss: 总优化损失
    """
    total_pho_loss = 0.0
    # 计算所有关键帧的光度损失
    for idx in range(len(poses)):
        rendered_img = render_gaussians(gaussians, poses[idx])  # 3DGS渲染
        total_pho_loss += photometric_loss(rendered_img, gt_imgs[idx])
    # 计算各向同性正则化损失（公式10）
    s_list = [gauss.scale for gauss in gaussians]  # 高斯缩放参数 s_i
    avg_s = torch.mean(torch.stack(s_list), dim=0)
    iso_loss = sum([torch.norm(s - avg_s, p=1) for s in s_list])
    # 总损失
    total_loss = total_pho_loss + lambda_iso * iso_loss
    return total_loss
```
