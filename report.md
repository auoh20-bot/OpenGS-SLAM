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
| (1) | 点图回归损失函数 | 内容 |  
| (4) | 光度损失函数 | 内容 |  
| (8) | 尺度比 | 内容 |  
| (9) | 帧间尺度因子 | 内容 |  
| (10) | 各向同性正则化损失 | 内容 |  
| (12) | 映射优化总损失 | 内容 |  
| (13) | 旋转弧度计算 | 内容 |  
| (14) | 动态迭代次数调整 | 内容 |  
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
 ![Image text](https://github.com/auoh20-bot/OpenGS-SLAM/blob/main/622d5aabee5d299e86f0685ea2c3735.png?raw=true)
 ![Image text]()
### 体积表示
体积表示已展现出一致重建的潜力，因其几何参数在渲染过程中相互耦合。<u>多种 SLAM 系统采用神经场 [25] 和高斯泼溅 [18] 中的可微分渲染技术，适用于单目和 RGB‐D 相机。然而相较于替代方案，这些方法的实时性能滞后，且需要深度、额外二维先验或缓慢相机运动来约束解。</u>面向通用场景重建的3D先验最早将二维特征融合为三维体素网格，再解码为表面几何 [27, 40]。此类方法假设已知融合位姿，故不适用于联合跟踪与地图构建，且体积表示需消耗大量内存并依赖预定义分辨率。

### 双视图三维重建先验
最近，<u><font style="background-color:#FCE75A;">DUSt3R</font></u><u> 引入了一种新颖的双视图三维重建先验，</u>**<u>可在共同坐标系中输出两幅图像的稠密三维点云</u>**<u>。</u>相较于先前解决任务子问题的先验方法，DUSt3R 通过隐式推理对应关系、位姿、相机模型和稠密几何，直接提供双视图三维场景的伪测量。

后续方法 <u><font style="background-color:#FCE75A;">MASt3R</font></u><u>[21] </u>**<u>预测额外的逐像素特征</u>**<u>以改进定位和运动恢复结构的像素匹配[10]</u>。然而与所有先验方法类似，其预测在三维几何中仍可能存在不一致性和相关误差。因此 DUSt3R 和 MASt3R‐SfM 需通过大规模优化确保全局一致性，但时间复杂度无法随图像数量良好扩展。Spann3R[49] 通过微调 DUSt3R 直接将点云图流预测到全局坐标系，从而放弃后端优化，但必须维持有限的 token 内存，这可能导致大场景中的漂移。

<u>在本研究中，我们</u>**<u>构建了一个围绕“双视图三维重建先验”的稠密 SLAM 系统。系统仅需通用的中心相机模型</u>**<u>，无需任何内参先验；通过高效的点图匹配、跟踪与融合、回环检测以及全局优化，实时地把成对预测拉成大规模全局一致的稠密地图</u>。

> 总结一下，单视图先验、多视图先验和体积表示存在各自缺陷，但是最新提出的 DUSt3R 利用隐式的双视图三维重建先验规避了如 MVS 或光流多视图先验存在的问题。之前已有将 MASt3R 应用在 SfM 中，因此作者想要做的是将 MASt3R 应用在实时性更高的 SLAM 系统中。
>

## 创新点归纳
<font style="color:#DF2A3F;">（创新点归纳）</font>

+ 首个使用双视图三维重建先验<u> MASt3R[21]作为基础的实时 SLAM 系统</u>。
+ 用于<u>点云图匹配 (Pointmap Matching)</u>、<u>跟踪与局部融合 (Tracking and Pointmap Fusion)</u>、<u>图构建与闭环检测 (Graph Construction and Loop Closure) </u>、 <u>后端全局优化 (Backend Optimisation) </u>以及<u>重定位 (Relocalisation)</u>的高效技术。

> 基于 MASt3R 去更新 SLAM 中各个组件，括号内容对应后续要说明的章节
>

+ 一种最先进的、能够处理<u>通用</u>且<u>随时间变化</u>的相机模型的稠密 SLAM 系统。

> 这里的时变是指传统针孔相机模型发生相机的参数需要重新标定的情况，如相机焦距在拍摄过程中发生变化。
>

## Method
<u>DUSt3R</u> 接收一对图像** **![image](https://cdn.nlark.com/yuque/__latex/864e19c17f3d060821e5424d78969f3e.svg)，并输出三维点图![image](https://cdn.nlark.com/yuque/__latex/5c1953adbfcde030f8e6d4f154f003dd.svg)，以及对应的置信度图![image](https://cdn.nlark.com/yuque/__latex/04c696869f5b322c2becc052a0be4f3b.svg)。在 <u>MASt3R</u> 中，额外增加了一个头部（head）来输出用于匹配的 d 维特征：![image](https://cdn.nlark.com/yuque/__latex/1e83581c8d9972723511cd5b39a1924e.svg)，以及对应的置信度![image](https://cdn.nlark.com/yuque/__latex/0bb3455430a0ce973f4c5343e93633c1.svg)。我们将![image](https://cdn.nlark.com/yuque/__latex/16b89b1ef44fa579382389d2d0fe8394.svg)定义为 MASt3R 的前向传播函数，它输出上述所有内容。

> 这里的符号![image](https://cdn.nlark.com/yuque/__latex/7fc0e9ba98bd8abb3ee3e13a189a72ba.svg)表示在相机![image](https://cdn.nlark.com/yuque/__latex/036441a335dd85c838f76d63a3db2363.svg)坐标系中表示的图像![image](https://cdn.nlark.com/yuque/__latex/00eb1ac84cb04ae9942cd8d9cf8ca49f.svg)的三维点图。
>
> 根据 MASt3R-SLAM/thirdparty/mast3r/README.md 可知：
>
> + 置信度![image](https://cdn.nlark.com/yuque/__latex/3e2f26cc65e9b309efa11c948e05cf2e.svg)<font style="color:rgb(51, 51, 51);background-color:rgb(248, 248, 248);">的取值范围是 [1, +∞)</font>
>

以下代码是 <u>MASt3R</u> 的输出。其中 X 的维度 (4, B, H, W, 3)，可知，<u>同一图像中像素和三维点的对应关系是已知的</u>。

```python
# MASt3R-SLAM/mast3r_slam/mast3r_utils.py L82-L115
# MASt3R模型批量解码函数：为成对图像生成密集的3D点云和特征
# 输入两张图像的特征，输出每张图像每个像素对应的3D世界坐标
# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    # 获取批次大小
    B = feat_i.shape[0]
    # 初始化存储4种匹配组合结果的列表
    # X: 3D点云坐标, C: 置信度, D: 描述符, Q: 描述符置信度
    X, C, D, Q = [], [], [], []

    # 对批次中的每对图像进行处理
    for b in range(B):
        # 提取第b对图像的特征和位置编码
        feat1 = feat_i[b][None]  # 第一张图像的特征 (1, C, H, W)
        feat2 = feat_j[b][None]  # 第二张图像的特征 (1, C, H, W)
        pos1 = pos_i[b][None]    # 第一张图像的位置编码
        pos2 = pos_j[b][None]    # 第二张图像的位置编码

        # 执行双向推理：图像1→图像2 和 图像2→图像1
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])

        # 组织4种匹配组合的结果
        # res11: 图像1→图像1 (自匹配)
        # res21: 图像2→图像1 (跨匹配)
        # res22: 图像2→图像2 (自匹配)
        # res12: 图像1→图像2 (跨匹配)
        res = [res11, res21, res22, res12]

        # 从每个推理结果中提取关键输出
        # r["pts3d"][0]: 3D点云，每个像素有(x,y,z)坐标
        # r["conf"][0]: 重建置信度
        # r["desc"][0]: 特征描述符
        # r["desc_conf"][0]: 描述符置信度
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )

        # 将当前批次的4种组合结果堆叠
        X.append(torch.stack(Xb, dim=0))  # (4, H, W, 3) - 4种组合的3D点云
        C.append(torch.stack(Cb, dim=0))  # (4, H, W, 1) - 对应的置信度
        D.append(torch.stack(Db, dim=0))  # (4, H, W, D) - 特征描述符
        Q.append(torch.stack(Qb, dim=0))  # (4, H, W, 1) - 描述符置信度

    # 将所有批次结果在batch维度上堆叠
    # 从 (B, 4, H, W, C) 转换为 (4, B, H, W, C)
    X, C, D, Q = (
        torch.stack(X, dim=1),  # (4, B, H, W, 3)
        torch.stack(C, dim=1),  # (4, B, H, W, 1)
        torch.stack(D, dim=1),  # (4, B, H, W, D)
        torch.stack(Q, dim=1),  # (4, B, H, W, 1)
    )

    # 对输出进行下采样以减少计算量
    X, C, D, Q = downsample(X, C, D, Q)

    # 返回最终结果：X是最重要的输出，包含每个像素的3D世界坐标
    return X, C, D, Q
```

在训练 MASt3R 的部分数据有明确尺度单位，如米；但是部分数据没有尺度单位，这就会导致产生<u>尺度不一致</u>性。<u>为了解决这个问题，对位姿使用 </u>![image](https://cdn.nlark.com/yuque/__latex/8d6aeca6316153ebd42b581085ad98fc.svg)<u>进行更新</u>：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761552841361-04f6ffb3-bcb2-4ce4-bc53-dfdfa1240b16.png)

其中![image](https://cdn.nlark.com/yuque/__latex/12c3a80ec64f31628bbc037c152c9a04.svg)。

我们对相机模型仅假设为通用中心相机（generic central camera）[35]，即所有光线都通过唯一的相机中心。定义函数：![image](https://cdn.nlark.com/yuque/__latex/9a2d07a64e91b86ded2051e05d00d097.svg)它将点图 ![image](https://cdn.nlark.com/yuque/__latex/b3bf0c71902b7fbb215f74fc408bb6e2.svg) 归一化为单位光线（unit rays），因此每个点图都定义了自身的相机模型。<u>这使得能够以统一的方式处理</u>**<u>时变</u>**<u>相机模型（例如变焦）和</u>**<u>畸变</u>**。

> 时变指基于针孔相机模型的相机内参需要更新的情况，畸变是指畸变参数需要更新的情况。
>
> [35] **Why having 10,000 parameters in your camera model is better than twelve**：提出基于中心相机模型进行特征匹配的方法。
>

##### 这里对参考文献 [35] 射线定义做进一步说明：
射线![image](https://cdn.nlark.com/yuque/__latex/9a2d07a64e91b86ded2051e05d00d097.svg)可以理解为相机![image](https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg)坐标系下一三维点![image](https://cdn.nlark.com/yuque/__latex/ec98ba941ef051ee8ca999e8282e8f4f.svg)进行 L2 归一化的单位向量，即![image](https://cdn.nlark.com/yuque/__latex/7d222616924d534560ab5531147246eb.svg)。

```python
# MASt3R-SLAM/mast3r_slam/matching.py L25-L49
# 为迭代投影匹配算法准备数据的预处理函数
def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    # 获取输入张量的批次大小、高度和宽度
    b, h, w, _ = X11.shape
    # 获取张量所在的设备（CPU或GPU）
    device = X11.device

    # 构建光线图像（归一化后的3D点云）
    # 对输入的3D点进行L2归一化
    rays_img = F.normalize(X11, dim=-1)
    # 重新排列维度为(batch, channels, height, width)格式
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,c,h,w)
    # 计算光线图像的梯度（x和y方向）
    gx_img, gy_img = img_utils.img_gradient(rays_img)
    # 将原始光线图像与梯度信息拼接，形成包含梯度的光线图像
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    # 重新排列回(batch, height, width, channels)格式并确保内存连续
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  # (b,h,w,c)

    # 准备要投影的3D点
    # 将第二幅图像的3D点展平为向量形式
    X21_vec = X21.view(b, -1, 3)
    # 对3D点进行L2归一化
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    # 设置投影的初始猜测
    # 如果没有提供初始索引映射，则使用恒等映射
    if idx_1_to_2_init is None:
        # 重置为恒等映射：创建从0到h*w-1的连续索引序列
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    # 将线性索引转换为像素坐标
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    # 转换为浮点型以便后续优化
    p_init = p_init.float()

    # 返回预处理后的数据：光线图像、归一化3D点、初始投影猜测
    return rays_with_grad_img, pts3d_norm, p_init
```

### Pointmap Matching
_对应关系_是 SLAM 的一个基本组成部分，同时被跟踪和建图所需要。在这种情况下，给定来自 <u>MASt3R</u> 的点云图和特征，我们需要<u>找到两幅图像之间的像素匹配集合</u>，表示为![image](https://cdn.nlark.com/yuque/__latex/66c2c4995a6e8a91572cc3d38b966c1e.svg)。

> _对应关系_：这里的对应关系是指同一相机![image](https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg)坐标下图像![image](https://cdn.nlark.com/yuque/__latex/190a077310286086074db80e583b7e1e.svg)之间像素坐标的对应关系。
>
> MASt3R 只能获取两组三维点，通过投影公式可以得到其对应的两组像素点，<u>但是并无法获得这两组像素点之间的对应关系</u>。  
可能会有一个比较直接的想法，即这两组三维点均在相机![image](https://cdn.nlark.com/yuque/__latex/4dd004c812cd8d82d0efed94734dd4da.svg)坐标系下，那么这两组中共视的那部分三维点坐标<u>理应</u>相同，那么可以通过投影公式找到对应的像素坐标，从而得到匹配关系。但是 MASt3R 得到的共视的那部分三维点坐标并不完全相同，会存在一定的误差值，从而导致上述的方法失效。
>

朴素的暴力匹配具有二次复杂度，因为它是对所有可能的像素对进行全局搜索。为了避免这种情况，<u>DUSt3R</u> 在三维点上使用 k‐d 树；然而，构建过程不易并行化，并且如果点云图预测存在误差，3D 中的最近邻搜索会找到许多不准确的匹配。

在 <u>MASt3R</u> 中，从网络预测了额外的高维特征以实现更宽的基线匹配，并提出了由粗到细方案来处理全局搜索。然而，密集像素匹配的运行时间在秒级，而稀疏匹配仍然比 k‐d 树慢。

<u>我们不是专注于高效方法进行全局匹配搜索，而是从优化作为局部搜索中寻找灵感</u>。

与特征匹配相比，我们受到密集 SLAM 中常用的投影数据关联方法的启发。然而，这需要具有闭式投影的参数化相机模型，而我们的<u>唯一假设是每帧具有唯一的相机中心</u>（中心相机模型）。给定输出点云图![image](https://cdn.nlark.com/yuque/__latex/f1e532d342cf994ee61388fa8ef3745c.svg) ,我们可以利用光线![image](https://cdn.nlark.com/yuque/__latex/9a2d07a64e91b86ded2051e05d00d097.svg)构建![image](https://cdn.nlark.com/yuque/__latex/4cc47a8bde7499b3333b451c922a5640.svg)的通用相机模型。受缺乏闭式投影的通用相机校准方法 [32, 35] 的启发，我们通过迭代优化参考帧中的像素坐标![image](https://cdn.nlark.com/yuque/__latex/8f4b562467978432828ee0e9ebaf90b3.svg)来独立投影每个点![image](https://cdn.nlark.com/yuque/__latex/307520fb8ac92217d54cc372c7318025.svg)，以最小化射线误差：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761706504716-4450df48-0705-4ac9-bb2f-7267279ff9b6.png)

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1761553228857-112c2e38-3d8c-4def-aef6-234f376798f3.png)

**图 2**. 迭代投影匹配概述：给定 <u>MASt3R</u> 生成的两个点云图预测，参考点云图经归一化![image](https://cdn.nlark.com/yuque/__latex/9a2d07a64e91b86ded2051e05d00d097.svg)获得平滑的像素到射线映射。对于点云图![image](https://cdn.nlark.com/yuque/__latex/d6a4771738f5b76b3397a5b04321af2f.svg)中三维点![image](https://cdn.nlark.com/yuque/__latex/2adf48e1cf80858f0604d2b51eeed4b8.svg)的初始投影估计值![image](https://cdn.nlark.com/yuque/__latex/489b5eb4a753c39916bf7b3ca484ee35.svg)，通过迭代更新像素以最小化查询射线![image](https://cdn.nlark.com/yuque/__latex/50a4d21310a46a8359e2464f22c6a0b0.svg)与目标射线![image](https://cdn.nlark.com/yuque/__latex/bfb2c4a16636d3c92c6d49fbc7bda0f3.svg)之间的角度误差![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)。当找到实现最小误差的像素![image](https://cdn.nlark.com/yuque/__latex/4be2e96825512ded115f9cecd2bbef00.svg)后，即建立![image](https://cdn.nlark.com/yuque/__latex/00eb1ac84cb04ae9942cd8d9cf8ca49f.svg)与![image](https://cdn.nlark.com/yuque/__latex/dd46b6b1182d153c71fddc2d06311ca0.svg)间的像素对应关系。

> 将图 2 分为三个部分，分别用 2.x 进行表示：
>
> 图 2.1 为图像![image](https://cdn.nlark.com/yuque/__latex/00eb1ac84cb04ae9942cd8d9cf8ca49f.svg)：相机![image](https://cdn.nlark.com/yuque/__latex/2443fbcfeb7e85e1d62b6f5e4f27207e.svg)坐标系下图像![image](https://cdn.nlark.com/yuque/__latex/00eb1ac84cb04ae9942cd8d9cf8ca49f.svg)中像素从初始位置![image](https://cdn.nlark.com/yuque/__latex/489b5eb4a753c39916bf7b3ca484ee35.svg)迭代到最终位置![image](https://cdn.nlark.com/yuque/__latex/4be2e96825512ded115f9cecd2bbef00.svg)的过程。彩色背景代表了图像![image](https://cdn.nlark.com/yuque/__latex/00eb1ac84cb04ae9942cd8d9cf8ca49f.svg)中像素的射线场，表明各个像素对应的单位射线方向。
>
> 图 2.2 为示意图：它是进行匹配计算的核心视图，展示了需要最小化的角度差异![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)，以及两条关键射线：目标射线 ![image](https://cdn.nlark.com/yuque/__latex/bfb2c4a16636d3c92c6d49fbc7bda0f3.svg) 和查询射线 ![image](https://cdn.nlark.com/yuque/__latex/9e8268fb6fe0a7e2ebe866c5e04bbf66.svg)。
>
> 图 2.3 为图像![image](https://cdn.nlark.com/yuque/__latex/dd46b6b1182d153c71fddc2d06311ca0.svg)：表示目标图像![image](https://cdn.nlark.com/yuque/__latex/dd46b6b1182d153c71fddc2d06311ca0.svg)的实际外观。 它用虚线箭头连接到中间的视图，指明了匹配的最终目标图像。
>

图 2 展示以上过程，需注意的是最小化归一化向量间的欧几里得距离等价于最小化两条归一化光线间的夹角：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762152397405-21b62233-7ada-4202-aceb-808de73d6c5c.png)

```python
# MASt3R-SLAM/mast3r_slam/backend/src/matching_kernels.cu L193-L198
    // ===== 第三步：计算残差和代价函数 =====
    // 计算归一化射线与目标3D点方向的差值（残差）
    #pragma unroll
    for (int j=0; j<3; j++) {
      err[j] = r[j] - pts_3d_norm[b][n][j];
    }
    // 计算代价函数（残差平方和）
    float cost = err[0]*err[0] + err[1]*err[1] + err[2]*err[2];
```

> 从代码来看，实际上仍以![image](https://cdn.nlark.com/yuque/__latex/1b14e76199740469c36a175c6053c7a7.svg)作为代价函数，这里提出角度只是方便理解代价结果，如：
>
> cost = 0：两个向量完全平行（cosθ = 1），完美匹配
>
> cost = 2：两个向量完全相反（cosθ = -1），最差匹配
>

通过使用与 [35]，相似的非线性最小二乘形式，我们可以通过计算解析雅可比矩阵并使用 LevenbergMarquardt 方法求解,迭代地解出投影位置的更新量。这可以针对每个点单独进行，并且由于射线图像是平滑的，几乎所有的有效像素在 10 次迭代内收敛。在此过程结束时, 我们现在得到初始匹配![image](https://cdn.nlark.com/yuque/__latex/3c9351549e89937570459de1ecd5afe6.svg)。当没有投影![image](https://cdn.nlark.com/yuque/__latex/8f4b562467978432828ee0e9ebaf90b3.svg)的初始估计值时，例如在与新关键帧进行跟踪或匹配闭环边时，所有像素都通过_恒等映射_进行初始化。

```python
# MASt3R-SLAM/mast3r_slam/matching.py L25-L49
    # 设置投影的初始猜测
    # 如果没有提供初始索引映射，则使用恒等映射
    if idx_1_to_2_init is None:
        # 重置为恒等映射：创建从0到h*w-1的连续索引序列
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    # 将线性索引转换为像素坐标
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    # 转换为浮点型以便后续优化
    p_init = p_init.float()
```

> _恒等映射_：假设图像![image](https://cdn.nlark.com/yuque/__latex/190a077310286086074db80e583b7e1e.svg)中像素一一对应进行初始化。这里 idx_1_to_2_init 重置为恒等映射得到的形状为 (b, h*w)，其中 b 为 batch_size，h*w 为像素数目，即每一组数值为 [0, 1, ..., h*w - 1]，共 b 组。
>

<u>该特征匹配结果不受位姿估计值影响，因其完全依赖 MASt3R输出</u>。

---

### Tracking and Pointmap Fusion
SLAM 的一个关键组成部分是对当前帧位姿相对于地图的低延迟跟踪。<u>作为一个基于关键帧的系统，我们估计当前帧</u>![image](https://cdn.nlark.com/yuque/__latex/a9f5f4fbe7a963704dbe10de8eb3cea9.svg)<u> 和上一个关键帧</u>![image](https://cdn.nlark.com/yuque/__latex/8e3898d751a1e64c54ce94f58eb22a9f.svg)<u>之间的相对变换</u>![image](https://cdn.nlark.com/yuque/__latex/bded59ae244ebb0474740d71459ca117.svg)。为了提高效率，我们希望只使用网络的一次前向传播来估计变换。假设我们已经有了上一个关键帧的点云图估计![image](https://cdn.nlark.com/yuque/__latex/06726241030eac7975bcd9f72ce05a0a.svg)，我们需要![image](https://cdn.nlark.com/yuque/__latex/9d14f891459cea2a1fed5f6c526b4e35.svg)帧中的点来求解![image](https://cdn.nlark.com/yuque/__latex/bded59ae244ebb0474740d71459ca117.svg)。这些点可以通过![image](https://cdn.nlark.com/yuque/__latex/3b94c122bf66351019bbdbebeba3e799.svg)获得。求解位姿的一种直接方法是最小化 3D 点误差：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762152838780-9bb0ac42-3252-444d-a1a7-fb1ef7469169.png)

其中![image](https://cdn.nlark.com/yuque/__latex/5669fd5094029e35831ff96496cf46bf.svg)是 <u>MASt3R-SfM [10]</u> 中提出的匹配置信度权重。为了提高鲁棒性，除了 Huber 范数![image](https://cdn.nlark.com/yuque/__latex/8fda43aaae04e025c40c78d86c0a4295.svg)之外，还应用了基于匹配的权重：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762152911660-8d4466aa-9c2d-42cc-9677-25ed92aa6f04.png)

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L813-L1140
/**
 * @brief 点对齐核函数：计算点对点匹配的Hessian矩阵和梯度
 * @param Twc 相机位姿 [num_poses, 8]，世界到相机的变换
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引 [num_edges]
 * @param jj 边的终止节点索引 [num_edges]
 * @param idx_ii2_jj 从j到i的匹配索引 [num_edges, num_points]
 * @param valid_match 匹配有效性标志 [num_edges, num_points, 1]
 * @param Q 匹配质量分数 [num_edges, num_points, 1] <- 这里 Q 对应 q_{m,n}
 * @param Hs 输出Hessian块矩阵 [4, num_edges, 7, 7]
 * @param gs 输出梯度向量 [2, num_edges, 7]
 * @param sigma_point 点距离的标准差（信息矩阵权重）<- 对应 sigma^2，默认值 0.05
 * @param C_thresh 置信度阈值
 * @param Q_thresh 匹配质量阈值
 * 
 * 每个block处理一条边，每个线程处理多个点
 * 计算点对点对齐的残差、Jacobian、Hessian和梯度
 */
__global__ void point_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引
 
  // 共享内存存储位姿，减少全局内存访问
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿 Tij = Ti^(-1) * Tj
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij); // <- 获取相对位姿 Tij
                                                    // 这里三个量对应 平移、旋转、缩放
  }
 
  __syncthreads();
 
  // 局部变量（每个线程独立）
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 将Xj变换到相机i坐标系
 
  // 残差
  float err[3];
  float w[3];  // 权重
 
  // Jacobian矩阵（14 = 7+7，对应两个位姿）
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储，14x14矩阵需要14*(14+1)/2=105个元素）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用此变量
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_point_inv = 1.0/sigma_point;  // 信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0];  // 匹配是否有效
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;  // j中的点k对应i中的索引

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // 将点Xj变换到相机i的坐标系：Xj_Ci = Tij * Xj
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // 计算残差（点之间的3D差异）// <- 计算残差 err
    err[0] = Xj_Ci[0] - Xi[0];
    err[1] = Xj_Ci[1] - Xi[1];
    err[2] = Xj_Ci[2] - Xi[2];
 
    // 计算权重（基于置信度和Huber核）
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh) # Q_thresh = 1.5
      & (ci > C_thresh) # C_thresh = 0
      & (cj > C_thresh);

    // 使用置信度加权
    const float conf_weight = q; // <- q_{m,n}
    // const float conf_weight = q * ci * cj;  // 可选：使用所有置信度的乘积
    
    // 计算sqrt(weight)，用于鲁棒核函数 // <- q_{m,n} / sigma^2
    const float sqrt_w_point = valid ? sigma_point_inv * sqrtf(conf_weight) : 0;
 
    // 应用Huber鲁棒权重 // <- huber(q_{m,n} * err / sigma^2)
    w[0] = huber(sqrt_w_point * err[0]);
    w[1] = huber(sqrt_w_point * err[1]);
    w[2] = huber(sqrt_w_point * err[2]);
    
    // 将sigma权重加回（完整权重 = sigma^2 * Huber权重）
    // <- 这一步是为了方便后续雅可比矩阵计算，实际上公式上一步就结束了？？？
    const float w_const_point = sqrt_w_point * sqrt_w_point;
    w[0] *= w_const_point;
    w[1] *= w_const_point;
    w[2] *= w_const_point;
 
    // 计算Jacobian矩阵
    // 残差对位姿的导数：d(err)/d(T) = d(Xj_Ci - Xi)/d(T)
    
    // X坐标的Jacobian
    // 对位姿j的导数（在相机i坐标系中的表示）
    Ji[0] = 1.0;  // 对tx的导数
    Ji[1] = 0.0;  // 对ty的导数
    Ji[2] = 0.0;  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数
    Ji[4] = Xj_Ci[2];  // 对旋转wy的导数：z
    Ji[5] = -Xj_Ci[1];  // 对旋转wz的导数：-y
    Ji[6] = Xj_Ci[0];  // 对缩放s的导数：x

    // 转换到全局坐标系，得到对位姿i的Jacobian
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];  // Ji = -Jj（负号因为是对Ti求导）

    // 累加Hessian矩阵：H += J^T * w * J
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];  // 上三角存储
        l++;
      }
    }
 
    // 累加梯度向量：g += J^T * w * err
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // Y坐标的Jacobian
    Ji[0] = 0.0;
    Ji[1] = 1.0;  // 对ty的导数
    Ji[2] = 0.0;
    Ji[3] = -Xj_Ci[2];  // 对旋转wx的导数：-z
    Ji[4] = 0;  // 对旋转wy的导数
    Ji[5] = Xj_Ci[0];  // 对旋转wz的导数：x
    Ji[6] = Xj_Ci[1];  // 对缩放s的导数：y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // Z坐标的Jacobian
    Ji[0] = 0.0;
    Ji[1] = 0.0;
    Ji[2] = 1.0;  // 对tz的导数
    Ji[3] = Xj_Ci[1];  // 对旋转wx的导数：y
    Ji[4] = -Xj_Ci[0];  // 对旋转wy的导数：-x
    Ji[5] = 0;  // 对旋转wz的导数
    Ji[6] = Xj_Ci[2];  // 对缩放s的导数：z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }
 
 
  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];  // 线程0写回结果
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  // 归约Hessian矩阵（14x14，分为4个7x7块）
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        // 根据索引将Hessian写入对应的块
        if (n<7 && m<7) {
          // Hii块（左上）
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];  // 对称矩阵
        }
        else if (n >=7 && m<7) {
          // Hij和Hji块（非对角）
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          // Hjj块（右下）
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];  // 对称矩阵
        }
      }
 
      l++;
    }
  }
}

```

尽管 3D 点误差是合适的，但它很容易受到 _点云图预测误差 _的影响，因为<u>深度预测不一致</u>的情况相对频繁。 鉴于我们最终将所有预测融合到一个单独的点云图中取平均，跟踪中的误差会降低关键帧点云图的质量，而这些点云图也会在后端使用。

> _点云图预测误差_：这里是指 <u>MASt3R</u> 输出点云图![image](https://cdn.nlark.com/yuque/__latex/f1e532d342cf994ee61388fa8ef3745c.svg)产生的误差，这是由于 <u>MASt3R</u>  的点图预测（尤其深度估计）在不同帧中可能不一致。
>

通过再次利用点云图预测可以在中心相机假设下转换为射线的特性，我们可以计算方向射线误差（directional ray error）来代替，<u>这种误差对不正确的深度预测不那么敏感</u>。为了计算这个误差，我们简单地将公式 (4) 中的两点都进行归一化：     

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762153223560-5350cafe-d015-4b92-848c-9906c4f8ed8f.png)

这导致了一个类似于公式 (3) 中提到的、并在图 2 中所示的角度误差，不同之处在于我们现在拥有许多已知的对应关系，并希望找到能最小化规范射线与当前帧对应预测射线之间所有角度误差的位姿。

鉴于角度误差是有界的，基于射线的误差对于离群点具有鲁棒性。<u>我们还包含了一个误差项，它带有较小的权重，用于计算点到相机中心距离的差异。</u>这可以防止系统在纯旋转下退化，同时避免深度误差带来的显著偏差。

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L455-L723
/**
 * @brief 射线对齐核函数：计算归一化射线匹配的Hessian矩阵和梯度
 * @param Twc 相机位姿 [num_poses, 8]
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2_jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param Hs 输出Hessian块矩阵
 * @param gs 输出梯度向量
 * @param sigma_ray 射线方向的标准差
 * @param sigma_dist 距离的标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 匹配质量阈值
 * 
 * 优化射线方向（归一化）和距离，而不是直接优化3D点位置
 * 这对于尺度模糊的场景更加鲁棒
 */
__global__ void ray_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_ray,
    const float sigma_dist,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引
 
  // 共享内存存储位姿
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // 局部变量
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 变换到相机i坐标系的点
 
  // 残差（4维：3维射线方向 + 1维距离）
  float err[4];
  float w[4];  // 权重
 
  // Jacobian矩阵
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_ray_inv = 1.0/sigma_ray;  // 射线方向的信息矩阵权重
  const float sigma_dist_inv = 1.0/sigma_dist;  // 距离的信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // 归一化测量点（计算单位方向向量）
    const float norm2_i = squared_norm3(Xi);  // ||Xi||^2
    const float norm1_i = sqrtf(norm2_i);  // ||Xi||
    const float norm1_i_inv = 1.0/norm1_i;    
    
    // 归一化射线：ri = Xi / ||Xi||
    float ri[3];
    for (int i=0; i<3; i++) ri[i] = norm1_i_inv * Xi[i]; // <- 计算 Xi 的归一化射线 ri
 
    // 将点Xj变换到相机i的坐标系
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // 计算预测点的范数
    const float norm2_j = squared_norm3(Xj_Ci);  // ||Xj_Ci||^2
    const float norm1_j = sqrtf(norm2_j);  // ||Xj_Ci||
    const float norm1_j_inv = 1.0/norm1_j;

    // 归一化预测射线：rj_Ci = Xj_Ci / ||Xj_Ci|| 
    float rj_Ci[3];
    for (int i=0; i<3; i++) rj_Ci[i] = norm1_j_inv * Xj_Ci[i]; // <- 计算 Xj_Ci 的归一化射线 rj_Ci
 
    // 计算残差（射线方向差异 + 距离差异）<- 计算残差 err
    err[0] = rj_Ci[0] - ri[0];  // x方向射线误差
    err[1] = rj_Ci[1] - ri[1];  // y方向射线误差
    err[2] = rj_Ci[2] - ri[2];  // z方向射线误差
    err[3] = norm1_j - norm1_i;  // 距离误差 <- 引入的基于 距离 的误差项
 
    // 计算权重
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // 使用置信度加权
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;  // 可选：使用所有置信度的乘积
    
    // 计算sqrt(weight)
    const float sqrt_w_ray = valid ? sigma_ray_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_dist = valid ? sigma_dist_inv * sqrtf(conf_weight) : 0;
 
    // 应用Huber鲁棒权重
    w[0] = huber(sqrt_w_ray * err[0]);
    w[1] = huber(sqrt_w_ray * err[1]);
    w[2] = huber(sqrt_w_ray * err[2]);
    w[3] = huber(sqrt_w_dist * err[3]);
    
    // 将sigma权重加回
    const float w_const_ray = sqrt_w_ray * sqrt_w_ray;
    const float w_const_dist = sqrt_w_dist * sqrt_w_dist;
    w[0] *= w_const_ray;
    w[1] *= w_const_ray;
    w[2] *= w_const_ray;
    w[3] *= w_const_dist;
 
    // 计算Jacobian矩阵
    // 归一化操作的导数：d(P/||P||)/dP = (I - P*P^T/||P||^2) / ||P||
    const float norm3_j_inv = norm1_j_inv / norm2_j;  // 1 / ||P||^3
    const float drx_dPx = norm1_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm3_j_inv;  // d(rx)/d(Px)
    const float dry_dPy = norm1_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm3_j_inv;  // d(ry)/d(Py)
    const float drz_dPz = norm1_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm3_j_inv;  // d(rz)/d(Pz)
    const float drx_dPy = - Xj_Ci[0]*Xj_Ci[1]*norm3_j_inv;  // d(rx)/d(Py)
    const float drx_dPz = - Xj_Ci[0]*Xj_Ci[2]*norm3_j_inv;  // d(rx)/d(Pz)
    const float dry_dPz = - Xj_Ci[1]*Xj_Ci[2]*norm3_j_inv;  // d(ry)/d(Pz)
 
    // rx坐标的Jacobian
    Ji[0] = drx_dPx;  // 对tx的导数
    Ji[1] = drx_dPy;  // 对ty的导数
    Ji[2] = drx_dPz;  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数
    Ji[4] = rj_Ci[2];  // 对旋转wy的导数：z
    Ji[5] = -rj_Ci[1];  // 对旋转wz的导数：-y
    Ji[6] = 0.0;  // 对缩放的导数（归一化后与缩放无关）

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian矩阵
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度向量
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // ry坐标的Jacobian
    Ji[0] = drx_dPy;  // d(ry)/d(Px) = d(rx)/d(Py)
    Ji[1] = dry_dPy;  // 对ty的导数
    Ji[2] = dry_dPz;  // 对tz的导数
    Ji[3] = -rj_Ci[2];  // 对旋转wx的导数：-z
    Ji[4] = 0.0;  // 对旋转wy的导数
    Ji[5] = rj_Ci[0];  // 对旋转wz的导数：x
    Ji[6] = 0.0;  // 对缩放的导数
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // rz坐标的Jacobian
    Ji[0] = drx_dPz;  // d(rz)/d(Px) = d(rx)/d(Pz)
    Ji[1] = dry_dPz;  // d(rz)/d(Py) = d(ry)/d(Pz)
    Ji[2] = drz_dPz;  // 对tz的导数
    Ji[3] = rj_Ci[1];  // 对旋转wx的导数：y
    Ji[4] = -rj_Ci[0];  // 对旋转wy的导数：-x
    Ji[5] = 0.0;  // 对旋转wz的导数
    Ji[6] = 0.0;  // 对缩放的导数
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }


    // 距离坐标的Jacobian
    // d(||P||)/dP = P / ||P|| = 归一化向量
    Ji[0] = rj_Ci[0];  // 对tx的导数
    Ji[1] = rj_Ci[1];  // 对ty的导数
    Ji[2] = rj_Ci[2];  // 对tz的导数
    Ji[3] = 0.0;  // 对旋转wx的导数（距离与旋转无关）
    Ji[4] = 0.0;  // 对旋转wy的导数
    Ji[5] = 0.0;  // 对旋转wz的导数
    Ji[6] = norm1_j;  // 对缩放s的导数：||P|| * ds = ||P||
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[3] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    // 累加梯度
    for (int n=0; n<7; n++) {
      vi[n] += w[3] * err[3] * Ji[n];
      vj[n] += w[3] * err[3] * Jj[n];
    }
 
 
  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}
```

我们在迭代重加权最小二乘（IRLS）框架中，使用<u>高斯-牛顿法</u>有效地求解位姿的更新。我们计算射线和距离误差关于相对位姿![image](https://cdn.nlark.com/yuque/__latex/bded59ae244ebb0474740d71459ca117.svg)的微扰![image](https://cdn.nlark.com/yuque/__latex/e7ccb9bf589e539415d2ed8b202fb932.svg)的解析雅可比矩阵。我们将残差![image](https://cdn.nlark.com/yuque/__latex/48463facf2e6bdd4218e7c2352e13a54.svg)、雅可比矩阵![image](https://cdn.nlark.com/yuque/__latex/c7d4a415e25716066a99bbd38864d63f.svg)和权重 ![image](https://cdn.nlark.com/yuque/__latex/dc10020247da5f8307363dbc8d72fdc8.svg)堆叠到相应的矩阵中，并迭代地求解线性系统并更新位姿：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762153310828-dde00d5a-0ac6-4523-a26a-4be6ff8039f7.png)

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1140-L1228
/**
 * @brief 基于点对点匹配的Gauss-Newton优化（CUDA实现）
 * @param Twc 相机位姿 [num_poses, 8]
 * @param Xs 3D点坐标 [num_poses, num_points, 3]
 * @param Cs 置信度 [num_poses, num_points, 1]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param sigma_point 点距离标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * @param max_iter 最大迭代次数
 * @param delta_thresh 收敛阈值
 * @return 最后一次迭代的增量（用于调试）
 * 
 * 迭代优化相机位姿，最小化点对点的3D距离
 */
std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();  // 获取tensor的设备和类型选项
  const int num_edges = ii.size(0);  // 边数量
  const int num_poses = Xs.size(0);  // 位姿数量
  const int n = Xs.size(1);  // 点数量

  const int num_fix = 1;  // 固定的位姿数量（通常是第一帧）

  // 设置索引
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);  // 获取所有唯一的关键帧索引
  // 用于边构建的索引
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // 用于线性系统索引（固定第一帧）
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7;  // Sim(3)的维度

  // 初始化缓冲区
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);  // Hessian块矩阵
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);  // 梯度向量

  // 用于调试输出
  torch::Tensor dx;

  torch::Tensor delta_norm;  // 增量范数

  // Gauss-Newton迭代
  for (int itr=0; itr<max_iter; itr++) {

    // 调用kernel计算Hessian和梯度
    point_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_point, C_thresh, Q_thresh
    );


    // 构建稀疏线性系统：位姿×位姿块
    SparseBlock A(num_poses - num_fix, pose_dim);

    // 更新Hessian矩阵（左端项）
    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    // 更新梯度向量（右端项）
    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // 求解线性系统：A*dx = -b
    // 注意：这里考虑了负号，因为求解的是下降方向
    dx = -A.solve();
    
    // 在Sim(3)流形上应用增量
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // 检查终止条件
    // 需要指定第二个参数，否则函数调用会有歧义
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;  // 增量足够小，收敛
    }
        

  }

  return {dx};  // 返回最后一次迭代的增量（用于调试）
}
```

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L725-L811
/**
 * @brief 基于射线对齐的Gauss-Newton优化（CUDA实现）
 * @param Twc 相机位姿
 * @param Xs 3D点坐标
 * @param Cs 置信度
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param sigma_ray 射线方向标准差
 * @param sigma_dist 距离标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * @param max_iter 最大迭代次数
 * @param delta_thresh 收敛阈值
 * @return 最后一次迭代的增量
 * 
 * 优化归一化射线方向和距离，适用于尺度模糊场景
 */
std::vector<torch::Tensor> gauss_newton_rays_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();  // 获取tensor选项
  const int num_edges = ii.size(0);  // 边数量
  const int num_poses = Xs.size(0);  // 位姿数量
  const int n = Xs.size(1);  // 点数量

  const int num_fix = 1;  // 固定位姿数量

  // 设置索引
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    //
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // 返回调试信息
}

```

由于每个点云图都可能提供有价值的新信息，我们利用这一点，不仅对几何估计进行滤波，还对相机模型本身进行滤波，因为它是由射线定义的。在求解出相对位姿后，我们可以使用变换![image](https://cdn.nlark.com/yuque/__latex/bded59ae244ebb0474740d71459ca117.svg)并通过运行 <u>加权平均滤波器</u>（running weighted average filter）更新规范点云图![image](https://cdn.nlark.com/yuque/__latex/1949fc953c4e93dc18b1de80cfe04cdc.svg)：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762153390684-63208b8c-b419-4f16-bd76-1608bad4ad17.png)

点云图最初由于只使用了_小基线（small baseline）_帧而具有较大的误差和较低的置信度，但滤波会融合来自多个视点的信息。我们试验了不同的更新规范点云图的方法，<u>发现加权平均最适合在滤除噪声的同时保持一致性</u>。与 MASt3R-SfM [10] 中的规范点云图相比，我们以增量方式计算此结果，并且需要对点进行变换，因为额外的网络预测![image](https://cdn.nlark.com/yuque/__latex/06726241030eac7975bcd9f72ce05a0a.svg)会减慢跟踪速度。<u>滤波在 SLAM 中拥有悠久的历史，其优势在于能够利用来自所有帧的信息，而无需来显式地优化所有的相机位姿，并在后端存储解码器（decoder）预测的所有点云图。</u>  

> _小基线（small baseline）_：两张图片间的相机位姿偏差较小。
>
> 从代码中可知，存在多种点云图更新方法，但是按照论文作者说法，加权平均的效果最好，代码中默认也启用的是加权平均。
>

```python
# MASt3R-SLAM/mast3r_slam/frame.py L74-L77
        # 模式5: "weighted_pointmap" - 在笛卡尔坐标系中按置信度加权融合
        elif filtering_mode == "weighted_pointmap":
            # 加权平均公式：X_new = (C_old * X_old + C_new * X_new) / (C_old + C_new)
            # 这样置信度高的观测会有更大的权重
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            # 累积置信度（用于后续加权计算）
            self.C = self.C + C
            # 增加更新计数
            self.N += 1
```

---

### Graph Construction and Loop Closure
在跟踪过程中，<u>如果有效匹配的数量或</u>![image](https://cdn.nlark.com/yuque/__latex/21cb014a5397e1eab4b24f3251e6ffad.svg)_<u>中独有的关键帧像素数量</u>_<u>低于阈值</u>![image](https://cdn.nlark.com/yuque/__latex/243867cb3ebf82b93de2cc577095f2a0.svg)<u>，则添加一个新的关键帧</u>![image](https://cdn.nlark.com/yuque/__latex/039d08dd1755571980718bb537d92f9a.svg)。在添加![image](https://cdn.nlark.com/yuque/__latex/039d08dd1755571980718bb537d92f9a.svg)之后，一条 双向边（bidirectional edge）会被添加到边列表![image](https://cdn.nlark.com/yuque/__latex/201c332d65d99168e5a95c980d8c5e83.svg)中，连接到上一个关键帧![image](https://cdn.nlark.com/yuque/__latex/1c7ad26313d77780167c0b8ca4ee6cf5.svg)。这在时间上顺序地约束了估计的位姿；然而，漂移（drift）仍然可能发生。

> ![image](https://cdn.nlark.com/yuque/__latex/21cb014a5397e1eab4b24f3251e6ffad.svg)_中独有的关键帧像素数量_：当前帧的有效匹配中，有多少像素比例对应到关键帧的不同（唯一）像素位置。
>

```python
# MASt3R-SLAM/mast3r_slam/tracker.py L103-L110
        # 关键帧选择：判断是否需要添加新的关键帧
        # valid_kf 形状为 [H, W]，布尔值，表示该像素位置是否满足关键帧选择条件
        n_valid = valid_kf.sum()  # 有效匹配的数量
        match_frac_k = n_valid / valid_kf.numel()  # 有效匹配比例（匹配数量指标）

        # 计算唯一关键帧像素的比例（唯一像素数量指标）
        # idx_f2k[valid_match_k[:, 0]] 获取所有有效匹配对应的关键帧像素索引
        ## idx_f2k 的形状是 [H, W, 2]，idx_f2k[i, j] = [u, v] 表示：
        ### 当前帧的像素位置 (i, j) 匹配到关键帧的像素位置 (u, v)
        # torch.unique() 计算唯一索引的数量
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        # 如果匹配比例或唯一像素比例低于阈值，需要添加新关键帧
        # 使用min()确保两个指标都要满足（论文中的ωk阈值）
        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
```

为了闭合（解决）小型和大型的回环（loops），我们采用了 MASt3R-SfM [10] 使用的 _聚合选择性匹配核（Aggregated Selective Match Kernel, ASMK）_[46, 47] 框架，该框架用于从编码特征中检索图像。虽然这个方法以前是在所有图像都可用的批处理设置（batch setting）中使用，但我们对其进行了修改，使其能够增量地工作。

我们使用![image](https://cdn.nlark.com/yuque/__latex/039d08dd1755571980718bb537d92f9a.svg)的编码特征查询数据库，以获得得分最高的![image](https://cdn.nlark.com/yuque/__latex/38a3f4d664b7a723d138f9d57be0c783.svg)张图像。由于码本（codebook）只有数万个中心点（centroids），我们发现进行密集的 <u>L2 距离</u>计算足以对特征进行量化。如果<u>检索分数高于阈值</u>![image](https://cdn.nlark.com/yuque/__latex/943d96ebb9125260533a4fc5fcd31b9f.svg)<u>，我们将这些图像对传递给 MASt3R 解码器</u>，并且如果<u>匹配的数量（如 3.2 节所述）高于阈值</u>![image](https://cdn.nlark.com/yuque/__latex/406f8f46e0376349569d311afafececb.svg)<u>，我们就添加双向边</u>。最后，我们<u>将新关键帧的编码特征添加到 </u>_<u>倒排文件索引（inverted file index）</u>_<u>中，从而更新检索数据库</u>。

> _倒排文件索引（inverted file index）_：一种数据结构，用于快速检索包含特定视觉词（visual words）的图像。
>

```python
# MASt3R-SLAM/mast3r_slam/retrieval_database.py L96-L105
    def quantize_custom(self, qvecs, params):
        """
        将查询向量量化到码本质心。
        使用高效的L2距离计算技巧，避免显式计算差值矩阵。
        
        输入:
            qvecs (torch.Tensor): 查询向量（2D张量，每行是一个特征向量）
            params (dict): 量化参数，包含"quantize"键，其下有"multiple_assignment"参数
        
        作用:
            计算查询向量与所有码本质心的L2距离，并找到每个查询向量的top-k个最近质心。
            使用数学技巧：||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T 来高效计算距离。
        
        输出:
            topk.indices (torch.Tensor): 每个查询向量对应的top-k个最近质心的索引
        """
        # 使用数学技巧高效计算L2距离矩阵，避免显式形成差值矩阵
        # ||qvec - centroid||^2 = ||qvec||^2 + ||centroid||^2 - 2*qvec*centroid^T
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]  # 每个查询向量的平方和，形状为[n, 1]
            + torch.sum(self.centroids**2, dim=1)[None, :]  # 每个质心的平方和，形状为[1, m]
            - 2 * (qvecs @ self.centroids.mT)  # 2倍的点积，形状为[n, m]
        )
        # 获取多重分配参数k（每个向量分配给k个最近的质心）
        k = params["quantize"]["multiple_assignment"]
        # 找到每个查询向量的top-k个最近质心（largest=False表示找最小值）
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        # 返回质心索引（不返回距离值）
        return topk.indices
```

```python
# MASt3R-SLAM/mast3r_slam/retrieval_database.py L65-L72
            # 过滤：只保留分数大于阈值的图像
            valid = topk_images.values > min_thresh
            # 获取满足条件的图像索引
            topk_image_inds = topk_images.indices[valid]
            # 转换为Python列表
            topk_image_inds = topk_image_inds.tolist()

        # 如果需要在查询后添加到数据库
        if add_after_query:
            # 将当前帧的特征添加到数据库
            self.add_to_database(feat_np, id_np, topk_codes)

        # 返回满足条件的top-k个相似关键帧索引
        return topk_image_inds
```

```python
# MASt3R-SLAM/mast3r_slam/global_opt.py L73-L80
        # 计算每个关键帧对的匹配比例：有效匹配数 / 总像素数
        # match_frac_j: 从i到j方向的匹配比例 [batch]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        # match_frac_i: 从j到i方向的匹配比例 [batch]
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        # ========== 步骤7: 转换为张量并检查边有效性 ==========
        # 将关键帧索引列表转换为PyTorch张量，便于后续的向量化操作
        ii_tensor = torch.as_tensor(ii, device=self.device)  # 源关键帧索引 [num_edges]
        jj_tensor = torch.as_tensor(jj, device=self.device)  # 目标关键帧索引 [num_edges]

        # 检查边的有效性：需要两个方向的匹配比例都满足阈值
        # NOTE: 要求两个方向的匹配比例都大于阈值，才接受这条边
        # 使用min()确保两个方向都要满足条件，这样能保证双向匹配的质量
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac  # 无效边掩码 [num_edges]
```

---

### Backend Optimisation
给定关键帧位姿![image](https://cdn.nlark.com/yuque/__latex/5c27c9d0da95777f868c62217668f0c8.svg)和规范点云图![image](https://cdn.nlark.com/yuque/__latex/4f300f007f7830edbb858495a34ba360.svg)的当前估计值，后端优化的目标是实现所有位姿和几何的全局一致性（global consistency）。

虽然以前的公式在每次迭代后都使用一阶优化并需要重新缩放 [10, 50]，但我们引入了一种高效的二阶优化方案。该方案通过固定第一个 7-自由度 Sim(3) 位姿来处理 量尺自由度（gauge freedom）问题。

我们联合最小化图中所有边![image](https://cdn.nlark.com/yuque/__latex/201c332d65d99168e5a95c980d8c5e83.svg)的射线误差：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762153752387-3f077bb4-d7de-4cd6-a598-81544d869f72.png)

其中![image](https://cdn.nlark.com/yuque/__latex/8ff813e5a45c00ebe23c87c03c4d83c3.svg)。给定![image](https://cdn.nlark.com/yuque/__latex/459f3c80a50b7be28751b0869ef5386a.svg)个关键帧，公式 (9) 形成并将![image](https://cdn.nlark.com/yuque/__latex/93d17bd5f63802832e74472fe63c78e9.svg)的块累积到![image](https://cdn.nlark.com/yuque/__latex/caff1c70c0becdce46503705eeb78fcf.svg)的 Hessian 矩阵中。我们再次使用类似于公式 (7) 的高斯-牛顿法来解决这个问题，但由于系统不密集，我们采用稀疏 Cholesky 分解。

Hessian 矩阵的构建是通过使用解析雅可比矩阵和并行归约来实现的，所有这些都在 CUDA 中实现。此外，我们再次添加了一个小的距离一致性误差项，以避免在纯旋转中出现退化。

对于每个新的关键帧，我们最多执行 10 次高斯-牛顿迭代，优化在收敛时终止。二阶信息极大地加快了全局优化速度，而我们高效的实现确保它不是整个系统的性能瓶颈。

```python
# MASt3R-SLAM/mast3r_slam/global_opt.py L129-L166
def solve_GN_rays(self):
        """
        功能: 使用无标定的射线几何误差，对因子图中的Sim(3)关键帧位姿进行二阶（Gauss-Newton）优化，
             固定前 pin 帧（通常为1帧）以消除尺度与整体自由度（gauge），并在收敛或达到迭代上限时结束。

        输入:
            无（使用类成员: self.frames, self.cfg, self.model 等；内部从因子图状态收集边与点/置信度等）

        主要步骤:
            1) 收集当前因子图的唯一关键帧索引并检查是否达到可优化规模（> pin）。
            2) 根据唯一关键帧索引提取其 canonical 点云 Xs、当前位姿 T_WCs 以及平均置信度 Cs。
            3) 将已有的单向边扩展为双向边，得到 ii, jj, 匹配索引与有效掩码以及匹配质量 Q。
            4) 从配置读取 GN 的各类权重/阈值与迭代/收敛参数。
            5) 调用后端 CUDA GN（gauss_newton_rays）构建稀疏Hessian并用稀疏Cholesky迭代求解。
            6) 将优化后的位姿（除固定的 pin 帧外）写回帧管理器。

        输出:
            无显式返回值。副作用是更新 self.frames 中对应关键帧的 T_WC。
        """
        pin = self.cfg["pin"]  # 固定的关键帧数量（用于消除gauge自由度）
        unique_kf_idx = self.get_unique_kf_idx()  # 当前图中涉及到的唯一关键帧索引（升序）
        n_unique_kf = unique_kf_idx.numel()  # 唯一关键帧数量
        if n_unique_kf <= pin:  # 若可优化的自由度不足（都被固定），则直接返回
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)  # 提取 canonical 点、当前位姿与平均置信度

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()  # 将单向边拓展为双向边并整理观测

        C_thresh = self.cfg["C_conf"]  # 置信度阈值
        Q_thresh = self.cfg["Q_conf"]  # 匹配质量阈值
        max_iter = self.cfg["max_iters"]  # GN 最大迭代次数
        sigma_ray = self.cfg["sigma_ray"]  # 射线方向项噪声
        sigma_dist = self.cfg["sigma_dist"]  # 射线距离项噪声
        delta_thresh = self.cfg["delta_norm"]  # 增量范数收敛阈值

        pose_data = T_WCs.data[:, 0, :]  # 提取 Sim(3) 参数向量（[N, 8]，t(3)+q(4)+s(1)）作为优化初值
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,       # 位姿参数（按 unique_kf_idx 对齐）
            Xs,              # 每个关键帧对应的 canonical 点云
            Cs,              # 每个关键帧对应的平均置信度（用于加权/筛选）
            ii,              # 边起点关键帧索引（图中的原索引）
            jj,              # 边终点关键帧索引（图中的原索引）
            idx_ii2jj,       # 像素/点匹配的索引映射（i->j）
            valid_match,     # 有效匹配掩码
            Q_ii2jj,         # 匹配质量分数（用于鲁棒加权）
            sigma_ray,       # 射线方向噪声
            sigma_dist,      # 射线距离噪声
            C_thresh,        # 置信度阈值
            Q_thresh,        # 质量阈值
            max_iter,        # 最大迭代次数
            delta_thresh,    # 收敛阈值
        )

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])  # 将固定帧之外的更新位姿写回
```

> 从代码中来看，上述的 solve_GN_rays 函数仅在后端优化和重定位时出现，这与原论文说在前端使用了该优化方法不符。？？
>

---

### Relocalisation
如果系统由于匹配数量不足而丢失跟踪，则会触发重定位（relocalisation）。对于一个新的帧，我们<u>使用更严格的分数阈值来查询检索数据库</u>。一旦<u>检索到的图像与当前帧有足够的匹配数量，该帧就会作为新的关键帧添加到图中</u>，并恢复跟踪。

```python
# MASt3R-SLAM/main.py L28-L71
def relocalization(frame, keyframes, factor_graph, retrieval_database):
    """
    功能: 尝试以当前帧为新关键帧，对数据库中召回的候选关键帧进行重定位匹配，
         若成功则将当前帧加入因子图并触发一次后端优化，以闭环/纠正漂移。

    输入:
        frame: Frame，当前帧（已包含图像、点云、位姿等）
        keyframes: SharedKeyframes，关键帧共享存储与并发控制
        factor_graph: FactorGraph，因子图对象，用于添加边并求解后端GN
        retrieval_database: 检索数据库（基于图像/特征），用于候选闭环召回

    输出:
        bool: 是否重定位成功（成功则会进行一次后端优化）
    """
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:  # 使用关键帧锁，确保并发安全
        kf_idx = []  # 候选关键帧索引集合
        retrieval_inds = retrieval_database.update(  # 基于数据库检索召回候选关键帧
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"], # <-使用更严格的分数阈值来查询检索数据库
        )
        kf_idx += retrieval_inds  # 合并候选
        successful_loop_closure = False  # 是否成功闭环
        if kf_idx:  # 存在候选才尝试
            keyframes.append(frame)  # 临时将当前帧作为关键帧加入（若失败会回滚）
            n_kf = len(keyframes)  # 关键帧数量
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)  # 与每个候选构建一条边
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(  # 尝试把边加入因子图（带质量/匹配比例过滤）
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"], # <-检索到的图像与当前帧有足够的匹配数量，该帧就会作为新的关键帧添加到图中
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(  # 只有重定位成功后才把该帧加入检索库
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()  # 将当前关键帧位姿初始化为候选位姿
            else:
                keyframes.pop_last()  # 回滚加入失败的关键帧
                print("Failed to relocalize")

        if successful_loop_closure:  # 闭环成功则触发一次后端优化
            if config["use_calib"]:
                factor_graph.solve_GN_calib()  # 有标定：像素+log深度GN
            else:
                factor_graph.solve_GN_rays()   # 无标定：射线几何GN
        return successful_loop_closure  # 返回是否成功
```

---

### Known Calibration
我们的系统可以在没有已知相机标定的情况下工作，但如果我们确实拥有标定参数，可以利用它进行两个直接的改变来提高精度。首先，在跟踪和建图（mapping）中用于优化的规范点云图，我们查询深度维度，并根据已知相机模型定义的射线来约束点云图，将其反投影（backprojected）。其次，我们改变优化中的残差，使其处于像素空间而不是射线空间。在后端，![image](https://cdn.nlark.com/yuque/__latex/4cc47a8bde7499b3333b451c922a5640.svg)中的一个像素![image](https://cdn.nlark.com/yuque/__latex/acdcf753ea739358818a2dce4593d70a.svg)是与它所匹配的 3D 点的投影进行比较的：

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762153939056-553c6a56-2ef0-46b7-94ba-c8c9ff31ad3a.png)

 其中![image](https://cdn.nlark.com/yuque/__latex/aa63813311a522aaa10a41976f257d1e.svg)是使用给定相机模型到像素空间的投影函数。此外，额外的距离残差（在之前的射线误差优化中用到）也被转换为深度以保持一致性。

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1231-L1543
/**
 * @brief 标定投影核函数：计算像素投影残差的Hessian矩阵和梯度
 * @param Twc 相机位姿
 * @param Xs 3D点坐标
 * @param Cs 置信度
 * @param K 相机内参矩阵 [3, 3]
 * @param ii 边的起始节点索引
 * @param jj 边的终止节点索引
 * @param idx_ii2_jj 匹配索引
 * @param valid_match 匹配有效性
 * @param Q 匹配质量
 * @param Hs 输出Hessian块矩阵
 * @param gs 输出梯度向量
 * @param height 图像高度
 * @param width 图像宽度
 * @param pixel_border 像素边界（忽略图像边缘的像素）
 * @param z_eps 深度阈值（避免负深度）
 * @param sigma_pixel 像素误差标准差
 * @param sigma_depth 深度误差标准差
 * @param C_thresh 置信度阈值
 * @param Q_thresh 质量阈值
 * 
 * 使用已知相机内参优化位姿，最小化像素重投影误差和深度误差
 */
__global__ void calib_proj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const int height,
    const int width,
    const int pixel_border,
    const float z_eps,
    const float sigma_pixel,
    const float sigma_depth,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc和Xs的第一维是位姿数量
  // ii, jj, Q的第一维是边数量
 
  const int block_id = blockIdx.x;  // 当前block处理的边ID
  const int thread_id = threadIdx.x;  // 线程ID
 
  const int num_points = Xs.size(1);  // 每个位姿的点数
 
  int ix = static_cast<int>(ii[block_id]);  // 边的起始位姿索引
  int jx = static_cast<int>(jj[block_id]);  // 边的终止位姿索引

  // 共享内存存储相机内参
  __shared__ float fx;  // 焦距x
  __shared__ float fy;  // 焦距y
  __shared__ float cx;  // 主点x
  __shared__ float cy;  // 主点y
 
  // 共享内存存储位姿
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  // 从全局内存加载相机内参到共享内存
  if (thread_id == 0) {
    fx = K[0][0];
    fy = K[1][1];
    cx = K[0][2];
    cy = K[1][2];
  }
 
  __syncthreads();
 
  // 从全局内存加载位姿到共享内存
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // 计算相对位姿
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // 局部变量
  float Xi[3];  // 位姿i处的点
  float Xj[3];  // 位姿j处的点
  float Xj_Ci[3];  // 变换到相机i坐标系的点
 
  // 残差（3维：2维像素误差 + 1维深度误差）
  float err[3];
  float w[3];  // 权重
 
  // Jacobian矩阵
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];  // 对位姿i的Jacobian
  float* Jj = &Jx[7];  // 对位姿j的Jacobian
 
  // Hessian矩阵（上三角存储）
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];  // 梯度向量
 
  int l; // 稍后在Hessian填充时重用
  // 初始化Hessian为0
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  // 初始化梯度为0
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // 参数
  const float sigma_pixel_inv = 1.0/sigma_pixel;  // 像素误差的信息矩阵权重
  const float sigma_depth_inv = 1.0/sigma_depth;  // 深度误差的信息矩阵权重
 
  __syncthreads();
 
  // 每个线程遍历部分点
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // 获取点对应关系
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    // 读取位姿i处的点
    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    // 读取位姿j处的点
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];

    // 获取测量像素坐标（从线性索引转换）
    const int u_target = ind_Xi % width;  // 目标像素u
    const int v_target = ind_Xi / width;  // 目标像素v
 
    // 将点Xj变换到相机i的坐标系
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // 检查点是否在相机前方（深度为正）
    const bool valid_z = ((Xj_Ci[2] > z_eps) && (Xi[2] > z_eps));

    // 处理深度相关变量（避免除零）
    const float zj_inv = valid_z ? 1.0/Xj_Ci[2] : 0.0;  // 深度倒数
    const float zj_log = valid_z ? logf(Xj_Ci[2]) : 0.0;  // log深度（预测）
    const float zi_log = valid_z ? logf(Xi[2]) : 0.0;  // log深度（测量）

    // 将3D点投影到像素平面
    const float x_div_z = Xj_Ci[0] * zj_inv;  // X/Z
    const float y_div_z = Xj_Ci[1] * zj_inv;  // Y/Z
    const float u = fx * x_div_z + cx;  // u = fx * X/Z + cx
    const float v = fy * y_div_z + cy;  // v = fy * Y/Z + cy

    // 检查投影是否在图像范围内（排除边界）
    const bool valid_u = ((u > pixel_border) && (u < width - 1 - pixel_border));
    const bool valid_v = ((v > pixel_border) && (v < height - 1 - pixel_border));

    // 计算残差（像素重投影误差 + 对数深度误差）
    err[0] = u - u_target;  // u方向像素误差
    err[1] = v - v_target;  // v方向像素误差
    err[2] = zj_log - zi_log;  // log深度误差（相对深度）

    // 计算权重
    const float q = Q[block_id][k][0];  // 匹配质量
    const float ci = Cs[ix][ind_Xi][0];  // 点i的置信度
    const float cj = Cs[jx][k][0];  // 点j的置信度
    const bool valid =
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & valid_u & valid_v & valid_z;  // 检查图像和深度有效性
    
    // 使用置信度加权
    const float conf_weight = q;
    
    // 计算sqrt(weight)
    const float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_depth = valid ? sigma_depth_inv * sqrtf(conf_weight) : 0;

    // 应用Huber鲁棒权重
    w[0] = huber(sqrt_w_pixel * err[0]);
    w[1] = huber(sqrt_w_pixel * err[1]);
    w[2] = huber(sqrt_w_depth * err[2]);
    
    // 将sigma权重加回
    const float w_const_pixel = sqrt_w_pixel * sqrt_w_pixel;
    const float w_const_depth = sqrt_w_depth * sqrt_w_depth;
    w[0] *= w_const_pixel;
    w[1] *= w_const_pixel;
    w[2] *= w_const_depth;

    // 计算Jacobian矩阵
    // 投影函数的导数：d(u,v)/d(X,Y,Z)

    // u坐标的Jacobian（像素u对3D点的导数）
    Ji[0] = fx * zj_inv;  // du/dX = fx/Z
    Ji[1] = 0.0;  // du/dY = 0
    Ji[2] = -fx * x_div_z * zj_inv;  // du/dZ = -fx*X/Z^2
    Ji[3] = -fx * x_div_z * y_div_z;  // du/dωx（旋转）
    Ji[4] = fx * (1 + x_div_z*x_div_z);  // du/dωy
    Ji[5] = -fx * y_div_z;  // du/dωz
    Ji[6] = 0.0;  // du/ds（缩放，投影后与缩放无关）

    // 转换到全局坐标系，得到对位姿i的Jacobian
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];  // Ji = -Jj（负号因为是对Ti求导）


    // 累加Hessian矩阵：H += J^T * w * J（上三角存储）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];  // 权重w[0]对应u方向误差
        l++;
      }
    }

    // 累加梯度向量：g += J^T * w * err
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];  // 位姿i的梯度
      vj[n] += w[0] * err[0] * Jj[n];  // 位姿j的梯度
    }

    // v坐标的Jacobian（像素v对3D点的导数）
    Ji[0] = 0.0;  // dv/dX = 0
    Ji[1] = fy * zj_inv;  // dv/dY = fy/Z
    Ji[2] = -fy * y_div_z * zj_inv;  // dv/dZ = -fy*Y/Z^2
    Ji[3] = -fy * (1 + y_div_z*y_div_z);  // dv/dωx（旋转）
    Ji[4] = fy * x_div_z * y_div_z;  // dv/dωy
    Ji[5] = fy * x_div_z;  // dv/dωz
    Ji[6] = 0.0;  // dv/ds

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian（v方向）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];  // 权重w[1]对应v方向误差
        l++;
      }
    }

    // 累加梯度（v方向）
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }

    // 深度坐标的Jacobian（log深度对3D点的导数）
    // d(log(Z))/d(X,Y,Z) = (0, 0, 1/Z)
    Ji[0] = 0.0;  // d(log Z)/dX = 0
    Ji[1] = 0.0;  // d(log Z)/dY = 0
    Ji[2] = zj_inv;  // d(log Z)/dZ = 1/Z
    Ji[3] = y_div_z;  // d(log Z)/dωx：Y/Z
    Ji[4] = -x_div_z;  // d(log Z)/dωy：-X/Z
    Ji[5] = 0.0;  // d(log Z)/dωz
    Ji[6] = 1.0;  // d(log Z)/ds：log深度随缩放线性变化

    // 转换到全局坐标系
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    // 累加Hessian（深度方向）
    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];  // 权重w[2]对应深度误差
        l++;
      }
    }

    // 累加梯度（深度方向）
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }

  }  // 结束点循环
 
  __syncthreads();
 
  // 使用block归约将所有线程的结果汇总
  __shared__ float sdata[THREADS];
  // 归约梯度向量
  for (int n=0; n<7; n++) {
    // 归约位姿i的梯度
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];  // 线程0写回结果
    }
 
    __syncthreads();
 
    // 归约位姿j的梯度
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  // 归约Hessian矩阵（14x14，分为4个7x7块）
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        // 根据索引将Hessian写入对应的块
        if (n<7 && m<7) {
          // Hii块（左上）
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];  // 对称矩阵
        }
        else if (n >=7 && m<7) {
          // Hij和Hji块（非对角）
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          // Hjj块（右下）
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];  // 对称矩阵
        }
      }
 
      l++;
    }
  }
}
```

```python
# MASt3R-SLAM/mast3r_slam/backend/src/gn_kernels.cu L1546-L1638
/**
* @brief 基于标定投影的Gauss-Newton优化（CUDA实现）
* @param Twc 相机位姿
* @param Xs 3D点坐标
* @param Cs 置信度
* @param K 相机内参矩阵
* @param ii 边的起始节点索引
* @param jj 边的终止节点索引
* @param idx_ii2jj 匹配索引
* @param valid_match 匹配有效性
* @param Q 匹配质量
* @param height 图像高度
* @param width 图像宽度
* @param pixel_border 像素边界
* @param z_eps 深度阈值
* @param sigma_pixel 像素标准差
* @param sigma_depth 深度标准差
* @param C_thresh 置信度阈值
* @param Q_thresh 质量阈值
* @param max_iter 最大迭代次数
* @param delta_thresh 收敛阈值
* @return 最后一次迭代的增量
* 
* 使用已知相机内参优化位姿，适用于已标定相机的SLAM系统
*/
std::vector<torch::Tensor> gauss_newton_calib_cuda(
    torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
torch::Tensor K,
torch::Tensor ii, torch::Tensor jj, 
torch::Tensor idx_ii2jj, torch::Tensor valid_match,
torch::Tensor Q,
const int height, const int width,
const int pixel_border,
const float z_eps,
const float sigma_pixel, const float sigma_depth,
const float C_thresh,
const float Q_thresh,
const int max_iter,
const float delta_thresh)
{
    auto opts = Twc.options();  // 获取tensor选项
const int num_edges = ii.size(0);  // 边数量
const int num_poses = Xs.size(0);  // 位姿数量
const int n = Xs.size(1);  // 点数量

const int num_fix = 1;  // 固定位姿数量（通常固定第一帧作为参考）

// 设置索引映射
// 获取边中涉及的所有唯一关键帧
torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);

// 为边构建创建索引（pin=0，包含所有位姿）
std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
torch::Tensor ii_edge = inds[0];  // 重新索引后的起始节点
torch::Tensor jj_edge = inds[1];  // 重新索引后的终止节点

// 为线性系统索引创建索引（pin=num_fix，固定前num_fix个位姿）
// 这样可以将全局索引映射到优化变量的局部索引
std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
torch::Tensor ii_opt = inds_opt[0];  // 优化用的起始节点索引
torch::Tensor jj_opt = inds_opt[1];  // 优化用的终止节点索引

const int pose_dim = 7;  // Sim(3)位姿的维度（3平移+4旋转）

// 初始化缓冲区
// Hs存储4个7x7块矩阵：[Hii, Hij, Hji, Hjj]，每条边一组
torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
// gs存储2个梯度向量：[gi, gj]，每条边一组
torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

// 用于调试的输出变量
torch::Tensor dx;  // 位姿增量

torch::Tensor delta_norm;  // 增量范数，用于判断收敛

// Gauss-Newton迭代主循环
for (int itr=0; itr<max_iter; itr++) {

    // 步骤1：调用CUDA kernel计算Hessian矩阵和梯度向量
    // 每条边启动一个block，每个block有THREADS个线程
    calib_proj_kernel<<<num_edges, THREADS>>>(
        Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),  // 当前位姿估计
Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   // 3D点云
Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   // 置信度
K.packed_accessor32<float,2,torch::RestrictPtrTraits>(),    // 相机内参
ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),     // 边的起始节点
jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),     // 边的终止节点
idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),   // 点匹配索引
valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(), // 匹配有效性
Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),          // 匹配质量
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),         // 输出：Hessian
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),         // 输出：梯度
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh
    );


    // 步骤2：构建稀疏线性系统 H*dx = -g
    SparseBlock A(num_poses - num_fix, pose_dim);  // 创建稀疏块矩阵

    // 将各条边的Hessian块填充到大的稀疏矩阵中
    // 需要指定每个块在大矩阵中的位置(ii, jj)
    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}),  // 将4个块展平
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}),   // 行索引：[ii, ii, jj, jj]
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));  // 列索引：[ii, jj, ii, jj]

    // 将各条边的梯度向量累加到大的梯度向量中
    A.update_rhs(gs.reshape({-1, pose_dim}),  // 将2个梯度展平
        torch::cat({ii_opt, jj_opt}));  // 节点索引：[ii, jj]

    // 步骤3：求解线性系统 H*dx = -g
    // 注意：这里加负号是因为我们要沿着梯度下降方向（-∇f）
    dx = -A.solve();

    
    // 步骤4：在Sim(3)流形上应用增量更新
    // Pose_new = exp(dx) * Pose_old
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),  // 位姿（输入输出）
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),   // 增量
      num_fix);  // 跳过前num_fix个固定位姿

    // 步骤5：检查终止条件
    // 计算增量的L2范数：||dx||
    // 需要明确指定第二个参数，否则会有函数重载歧义
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;  // 如果增量足够小，认为已收敛，提前终止迭代
    }
        

  }  // 结束迭代循环

  return {dx};  // 返回最后一次迭代的增量（用于调试和分析）
}
```

## Results
我们在多种真实世界数据集上评估了系统性能。针对定位任务，我们在 TUM RGB‐D [38]，7‐Scenes [36]，ETH3D‐SLAM [34], 和 EuRoC [3] 上评估单目SLAM（均采用单目 RGB 设置）。几何评估方面，选用提供 3D 结构扫描真值的EuRoC Vicon 房间序列，以及具备深度相机测量的 7‐Scenes 数据集。

我们在配备 Intel Core i9 12900K 3.50GHz 的台式机和单块英伟达 GeForce RTX 4090 上运行系统。由于系统以约 15 帧率运行，我们对数据集的<u>每 2 帧</u>进行子采样以模拟实时性能。注意，我们使用来自 MASt3R 的全分辨率输出，该输出将最大维度调整为 512 大小。

### Camera Pose Estimation
<u>对于所有数据集，我们报告以米为单位的绝对轨迹误差 (ATE) 的均方根误差 (RMSE)。由于所有系统均为单目，我们执行缩放轨迹对齐。我们将未使用已知标定的系统表示为 Ours*。</u>

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762823754261-3e786af6-89d1-4a03-b981-0c89001228e8.png)

**表 1：TUM RGB-D 数据集上 ATE 评估（单位 m）**

> **加粗**最优，<u>下划线</u>次优，标定 (Calibrated) 和未标定 (Uncalibrated) 的一起比较。
>

**TUM RGB‐D 数据集**：在 TUM 数据集上，我们展示了最先进的轨迹误差，当使用校准如_表 1_。许多先前表现最佳的算法，例如 DROID‐SLAM、DPV‐SLAM 和 GO‐SLAM，都基于 DROID‐SLAM 提出的基础匹配和端到端系统。相比之下，我们提出了一种独特的系统，该系统采用现成的双视图几何先验，并展示出它能够在实时运行时超越其他系统。

此外，我们的未校准系统显著超越了 一个基线（我们将其记为 DROID‐SLAM*），该基线使用 GeoCalib 来校准内参 [48] 在序列的首张图像上，随后被 DROID‐SLAM 使用。我们实现这一点而无需假设一个固定的相机模型跨越整个序列，并且证明了相对于那些解决子问题的先验，3D 先验对于稠密未校准 SLAM 更有价值，相对于那些解决子问题的先验。我们的未校准 SLAM 结果也与来自近期学习技术的结果可比拟，例如具有已知标定的 DPV‐SLAM。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762824062153-6d0a56dc-db69-464f-8587-ccbade7d57d9.png)

**表 2：7-Scenes 数据集上 ATE 评估（单位 m）**

**7-Scenes**：我们使用相同的序列进行评估，遵循 NICER‐SLAM，如_表 2_。我们校准后的系统（Our）超越了 NICER‐SLAM [58] 和 DROID‐SLAM。此外，我们实时的未校准系统（Ours*）使用单一的三维重建先验，超越了 NICER‐SLAM，其在深度、法线和光流网络中使用多个先验，并且离线运行。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762824270309-1ab2bc7f-0514-42c8-a10c-22912137d512.png)

**图 5：ETH3D-SLAM 数据集上 ATE 评估（单位 m）**

> **横轴 (ATE [m])**：表示系统重建轨迹与真实轨迹的平均位置误差（单位为米）。越小越好。    
**纵轴 (# Successful Datasets)**：表示在<u>误差小于某个 ATE 阈值</u>时，算法成功重建的数据集数量。越高越好。   
曲线越“靠左靠上”，代表算法越优秀。   
>

**ETH3D-SLAM**：由于该数据集的难度，ETH3D‐SLAM 此前仅针对 RGB‐D 方法进行了评估。由于官方私有评估的<u> ATE 阈值</u>对于单目方法过于严格，我们在训练序列上评估了几种最先进的单目系统，并生成了 ATE 曲线。该数据集包含具有快速相机运动的序列，因此，对于所有方法，我们不对帧进行子采样。尽管其他方法可能具有更精确的轨迹，但我们的方法在鲁棒性方面具有更长的尾部，从而在绝对轨迹误差和曲线下面积 (AUC) 上都取得了最佳结果。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762825055664-aa8ea3e8-e760-43be-a05c-80920a2a67f3.png)

**表 3：7-Scenes 数据集和 EuRoC 数据集上****<u>重建效果</u>****评估（单位 m）**

> **ATE**：表示相机轨迹的平均绝对误差，<u>数值越小</u>代表定位越准确。  
**Accuracy**（重建精度）：测量重建点云与真实点云之间的平均距离，<u>越小越好</u>。  
**Completion**（重建完整度）：衡量重建结果是否覆盖完整场景，<u>越小表示越完整</u>。  
**Chamfer**（倒角距离）：综合考虑重建精度与完整度的距离度量，<u>越小表示整体重建效果越好</u>。
>

**EuRoC**：我们报告了所有 11 个 EuRoC 序列的平均绝对轨迹误差，见_表 3_。在未校准情况下 (Ours*)，我们发现畸变过于严重，因为 MASt3R 尚未针对此类相机模型进行训练，因此我们对图像进行了去畸变处理，但未对流程的其余部分进行校准。总体而言，我们的系统性能不如 DROID‐SLAM，但 DROID‐SLAM 在训练中显式增加了 10% 的灰度图像。然而，0.041m 的绝对轨迹误差仍然非常精确，并且从比较中可以看出，所有性能优越的方法都建立在 DROID‐SLAM 的基础之上，而我们提出了一种使用三维重建先验的新方法。

> 这里只用看 _表 3_ 的 ATE 指标。
>

### Dense Geometry Evaluation
我们在 EuRoC Vicon 房间序列和 7‐Scenes seq‐01 上评估了我们的几何方法与 DROID‐SLAM 及 Spann3R [49] 的性能对比。对于 EuRoC，通过将估计轨迹与 Vicon 轨迹对齐，获得参考点云与估计点云之间的配准。需注意，这种设置对 DROID‐SLAM 有利，因其能获得更低的轨迹误差。 对于 7‐Scenes，我们使用数据集提供的位姿对深度图像进行反投影以创建参考点云，随后通过迭代最近点 (ICP) 将其与估计点云对齐——由于未提供 RGB 与深度传感器之间的外参标定。

我们报告精度的均方根误差 (RMSE)，其定义为每个估计点与其最近参考点之间的距离，以及完整性的均方根误差，即每个参考点与其最近估计点之间的距离。两项指标均在 0.5m 的最大距离阈值下计算，并在所有序列上取平均值。我们还报告了倒角距离（Chamfer Distance），即这两项指标的平均值。

_表 3_ 总结了在 7‐Scenes 和 EuRoC 上的几何评估结果。在 7‐Scenes 数据集上，无论是否进行校准,我们的方法以及Spann3R 都比 DROID‐SLAM 实现了更精确的重建，这凸显了 3D 先验的优势。我们在两种不同设置下运行 Spann3R：一种是每 20 帧图像选取一个关键帧，另一种是每 2 帧图像选取一个关键帧。两种设置的差异表明免测试时优化的方法在泛化性方面面临挑战。无校准的我们的方法在精度 (Accuracy) 和倒角距离 (Chambe) 指标上均表现最佳，这归因于 7‐Scenes 提供的默认出厂标定内参。

> 这里看_ 表 3 _的 Accuracy、Completion 和 Chamber 指标。
>

对于 EuRoC 数据集，由于序列不以物体为中心，Spann3R 表现不佳因此被排除。如_ 表3 _所示，尽管 DROID‐SLAM 在绝对轨迹误差指标 (ATE) 上优于我们的方法，但无论是否经过校准，我们的方法都获得了更好的几何重建质量。DROID‐SLAM 因估计了大量环绕参考点云的噪声点而获得更高完整性，但我们的方法具有显著更优的精度。值得注意的是，我们的未校准系统产生了明显更大的绝对轨迹误差 (ATE) ，但在倒角距离指标 (Chambe) 上仍优于 DROID‐SLAM。

### Qualitative Results
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762826142237-ff3c2e35-dbe1-4d8a-9a97-68a4a940dac2.png)

**图 1：《公民》序列重建（Uncalibrated）**

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762826297872-e107bfae-798e-4ede-bd9d-b5c63d8457c9.png)

**图 4：TUM fr1/floor 序列重建**

_图1_ 展示了镜面人物上可匹配特征较少的挑战性《公民》 序列的重建。我们在_图 4_ 和_图 6_ 中展示了 TUM 和 EuRoC 数据集的位姿估计和稠密重建示例。此外，我们在 _图 7_ 中展示了连续关键帧间极端变焦变化的示例。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762826345081-67029c79-c950-42cb-9d64-9534784ef44e.png)

**图 6：EuRoC Machine Hall 04 序列重建**

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762826380035-9c02070f-357f-4150-ae42-69a52791ba69.png)

**图 7：EuRoC Machine Hall 04 序列重建（Uncalibrated）**

### Component Analysis
![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762826805579-7e0e4f1c-df9c-44ca-be3c-d218a44135ca.png)

**表 4：匹配方法比较**

> 这里 k-d tree 是 DUSt3R 的匹配方法。
>

我们在_表 4 _中比较匹配方法。我们的并行投影匹配配合特征优化实现了最佳精度，且运行时间显著缩短。在整个像素上进行 MASt3R 匹配耗时 2 秒，而我们的匹配仅需 2 毫秒，使整个系统帧率提升近 40 倍。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762827030695-8d54dfff-609a-4a8b-a5d8-42f152e83d75.png)

**表 5：点云融合方法比较**

> 这几个方法在 MAST3R-SLAM/config/base.yaml 中 filtering_mode 参数可调，具体实现在 MAST3R-SLAM/mast3r_slam/frame.py 的 update_pointmap 函数中实现。
>

```python
# MASt3R-SLAM/mast3r_slam/retrieval_database.py L41-L108
    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        """
        更新关键帧的点云地图，根据配置的过滤模式融合新旧点云数据
        
        该函数用于将新的点云观测融合到关键帧的现有点云地图中。支持多种融合策略：
        - first: 仅使用第一次更新
        - recent: 总是使用最新的观测
        - best_score: 使用置信度评分最高的观测
        - indep_conf: 独立地按置信度选择每个点
        - weighted_pointmap: 在笛卡尔坐标系中按置信度加权融合
        - weighted_spherical: 在球坐标系中按置信度加权融合
        
        Args:
            X (torch.Tensor): 新的3D点云坐标 [H, W, 3] 或 [N, 3]，表示要融合的点坐标
            C (torch.Tensor): 新的置信度值 [H, W, 1] 或 [N, 1]，对应每个点的置信度
            
        Returns:
            None: 直接修改 self.X_canon（更新后的点云）和 self.C（累积的置信度）
        """
        # 从配置中获取点云融合模式
        filtering_mode = config["tracking"]["filtering_mode"]

        # 如果点云地图为空（首次初始化），直接使用新数据
        if self.N == 0:
            # 克隆新的点云坐标（避免修改原始数据）
            self.X_canon = X.clone()
            # 克隆新的置信度值
            self.C = C.clone()
            # 设置更新次数为1
            self.N = 1
            # 设置总更新次数为1
            self.N_updates = 1
            # 如果使用best_score模式，需要初始化评分
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        # 模式1: "first" - 仅保留第一次更新（第二次更新时）
        if filtering_mode == "first":
            # 只在第二次更新时（N_updates == 1表示第一次更新已完成，这是第二次）
            if self.N_updates == 1:
                # 使用新数据替换旧数据
                self.X_canon = X.clone()
                self.C = C.clone()
                # 重置更新计数
                self.N = 1
        # 模式2: "recent" - 总是使用最新的观测数据
        elif filtering_mode == "recent":
            # 直接替换为新数据
            self.X_canon = X.clone()
            self.C = C.clone()
            # 重置更新计数
            self.N = 1
        # 模式3: "best_score" - 使用置信度评分最高的观测
        elif filtering_mode == "best_score":
            # 计算新数据的置信度评分（中位数或均值）
            new_score = self.get_score(C)
            # 如果新数据的评分更高，则替换
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                # 更新保存的评分
                self.score = new_score
        # 模式4: "indep_conf" - 独立地按置信度选择每个点
        elif filtering_mode == "indep_conf":
            # 创建掩码：新置信度大于旧置信度的位置
            new_mask = C > self.C
            # 将掩码扩展到3个坐标维度（x, y, z）
            # repeat(1, 3) 表示在最后一个维度重复3次
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            # 更新置信度更高的位置
            self.C[new_mask] = C[new_mask]
            # 重置更新计数
            self.N = 1
        # 模式5: "weighted_pointmap" - 在笛卡尔坐标系中按置信度加权融合
        elif filtering_mode == "weighted_pointmap":
            # 加权平均公式：X_new = (C_old * X_old + C_new * X_new) / (C_old + C_new)
            # 这样置信度高的观测会有更大的权重
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            # 累积置信度（用于后续加权计算）
            self.C = self.C + C
            # 增加更新计数
            self.N += 1
        # 模式6: "weighted_spherical" - 在球坐标系中按置信度加权融合
        elif filtering_mode == "weighted_spherical":
            # 辅助函数：将笛卡尔坐标转换为球坐标 (x, y, z) -> (r, phi, theta)
            def cartesian_to_spherical(P):
                # r: 径向距离（点到原点的距离）
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                # 将点云分割为x, y, z三个分量
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                # phi: 方位角（在xy平面上的角度，范围[-π, π]）
                phi = torch.atan2(y, x)
                # theta: 极角（与z轴的角度，范围[0, π]）
                theta = torch.acos(z / r)
                # 组合为球坐标 (r, phi, theta)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            # 辅助函数：将球坐标转换为笛卡尔坐标 (r, phi, theta) -> (x, y, z)
            def spherical_to_cartesian(spherical):
                # 分割球坐标为r, phi, theta
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                # 根据球坐标公式计算笛卡尔坐标
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                # 组合为笛卡尔坐标
                P = torch.cat((x, y, z), dim=-1)
                return P

            # 将旧的笛卡尔坐标转换为球坐标
            spherical1 = cartesian_to_spherical(self.X_canon)
            # 将新的笛卡尔坐标转换为球坐标
            spherical2 = cartesian_to_spherical(X)
            # 在球坐标系中进行加权平均（与笛卡尔坐标系类似）
            # 这样可以在球坐标系中更好地融合方向和距离信息
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            # 将融合后的球坐标转换回笛卡尔坐标
            self.X_canon = spherical_to_cartesian(spherical)
            # 累积置信度
            self.C = self.C + C
            # 增加更新计数
            self.N += 1

        # 无论哪种模式，都增加总更新次数（用于统计）
        self.N_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None
```

在_表 5_ 中，我们测试了更新规范点云图的不同方法，并报告了 TUM、7‐Scenes 和 EuRoC 数据集的平均绝对轨迹误差 (ATE)。选择最近 (Recent) 的和最早 (First) 的点云图分别会导致漂移和基线不足。在校准的情况下，加权融合 (Weighted) 与选择具有最高中位数置信度 (Median) 的点云图表现相当，但在未校准的情况下，加权融合 (Weighted) 实现了最低的绝对轨迹误差，并将 EuRoC 上的绝对轨迹误差提高了 1.3cm，这表明融合相机模型非常重要。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762996410507-af09392f-01c4-4984-9b3f-735fce6d4a6f.png)

**表 6：三维点和光线误差公式 在 未校准系统 中 ATE 指标对比**

在_表 6 _中，针对未校准的（Uncalibrated）跟踪和后端优化，光线误差公式相比使用包含不准确深度预测的三维点误差，提升了性能。

![](https://cdn.nlark.com/yuque/0/2025/png/45861457/1762996641175-ce40ca80-7107-49ad-9180-1ae8c1286b70.png)

**表 7：有无回环下系统轨迹精度与重建效果评估**

_表 7_ 显示，闭环检测提高了位姿和几何精度，且在更长的序列上收益更为显著。这表明 MASt3R 的输出仍存在偏差并导致漂移，而我们的组件正是为缓解这一问题而设计。

## Limitations and Future Work
虽然我们可以通过在前端滤波点云图来估计精确几何，但目前我们<u>并未在完整的全局优化中优化所有几何</u>。虽然 DROID‐SLAM 通过光束法平差优化每像素深度，但该框架<u>允许</u>_<u>非连贯几何</u>_。一种能在 3D 中使点云图全局一致且保持原始 MASt3R 预测连贯性的方法，并且完全实时将是未来工作中一个有趣的方向。

> _非连贯几何_与_连贯几何_：   
DROID-SLAM 允许每个像素单独调整深度，因此优化后虽然误差最小，但整体表面可能变得“不光滑”，即几何结构不连贯；  
MASt3R 不是预测独立深度，而是预测 pointmap（点图），同时，这个点云是由 网络在 3D 空间中直接推理 出来的，而不是通过多视图几何反演得到， 所以 MASt3R 的输出是一种“全局连贯的几何预测”，即使没有进行全局优化（bundle adjustment），也能在视觉上保持物体结构的一致性。  
>

由于 MASt3R <u>目前仅在针孔图像</u>上进行训练，其几何预测的准确性会随着图像畸变的增加而下降。然而，<u>未来模型将在多种相机模型上进行训练</u>，并将兼容于我们的框架，该框架从不假设_参数化相机模型_。此外，在全分辨率下使用解码器（MASt3R 的 decoder）目前是一个瓶颈，尤其对于低延迟跟踪和检查闭环候选帧而言。提升网络吞吐量将有益于整体系统效率。

> _参数化相机模型_：这里的参数化相机模型是如针孔相机模型等有内参等含参的相机模型，而该论文中使用的（基于光线）的中心相机模型就不属于该列。
>
> MASt3R 的解码器（decoder）需要在全分辨率下工作，才能输出高质量的点图，这种全分辨率解码导致计算开销大、显存占用高，难以适配 SLAM 的实时性要求。
>

## Conclusion
我们提出了一种基于 MASt3R 的实时稠密 SLAM 系统，该系统处理野外视频并实现最先进的性能。SLAM 领域最近的许多进展都遵循了 DROID‐SLAM 的贡献，它训练了一个端到端框架，通过光流更新来求解位姿和几何。我们采用了一种不同的方法，围绕一个现成的几何先验构建系统，首次实现了可比较的位姿估计，同时提供了一致的稠密几何。

## 重要参考文献
> 仅选取重要部分
>

[35] **Why having 10,000 parameters in your camera model is better than twelve**：提出基于中心相机模型进行特征匹配的方法。  
[Schops_Why_Having_10000_Parameters_in_Your_Camera_Model_Is_Better_CVPR_2020_paper.pdf](https://leedong25.yuque.com/attachments/yuque/0/2025/pdf/45861457/1761700246166-69446bd2-3056-4bd6-8fb9-8bd34d93437e.pdf)  
[https://arxiv.org/abs/1912.02908](https://arxiv.org/abs/1912.02908)

# 项目部署与测试
## 项目部署
### 创建环境 && 安装依赖
```python
conda create -n mast3r-slam python=3.11
conda activate mast3r-slam
```

```python
nvcc --version
```

```python
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

```python
git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive
cd MASt3R-SLAM/

# 服务器docker环境无法翻墙，可考虑先在本地下载后将文件传输过去

pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .
 

# Optionally install torchcodec for faster mp4 loading
pip install torchcodec==0.1
```

这一步 pip install --no-build-isolation -e . 中有 "lietorch @ git+[https://github.com/princeton-vl/lietorch.git"](https://github.com/princeton-vl/lietorch.git")，由于 Docker 环境无法翻墙，需要源码安装 lietorch：  
**Step1 下载源码**：[https://github.com/princeton-vl/lietorch](https://github.com/princeton-vl/lietorch)，存放位置不指定，只需要 conda 环境正确即可

**Step2 安装 lietorch 所需依赖**：

```python
# install requirements（前面安装过可以跳过）
pip install torch torchvision torchaudio wheel

# optional: specify GPU architectures
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"

# install lietorch
pip install --no-build-isolation .
```

**Step3 重新运行指令**：运行之前需要将 "lietorch @ git+[https://github.com/princeton-vl/lietorch.git"](https://github.com/princeton-vl/lietorch.git") 这一行注释掉。

```python
pip install --no-build-isolation -e . 
```

```python
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
```

## 数据集下载
scripts 里面的 bash 脚本链接过时，需到官网手动下载，以 TUM RGBD 为例：

在官网 [https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) 下载 scripts/download_tum.sh 中所需数据集并解压，注意保持路径为 datasets/tum

