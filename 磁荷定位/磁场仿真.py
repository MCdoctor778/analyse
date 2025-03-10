import os
# 设置环境变量解决OpenMP错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import torch
import math
import warnings
from PIL import Image

# 忽略libpng警告
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# 1. 准备：基本常数和示例参数
# -----------------------------
mu0 = 4.0 * math.pi * 1e-7  # [H/m] 真空磁导率
z_obs = 100e-9             # [m] 观测平面，100 nm

# 假设我们有 N=3 个偶极子，位置在 z=0 平面上
dipole_positions = torch.tensor([
    [0.0,  0.0,  0.0],      # 第1个偶极子坐标
    [1e-7, 0.0,  0.0],      # 第2个偶极子坐标 (举例: x=100 nm)
    [0.0,  2e-7, 0.0],      # 第3个偶极子坐标 (举例: y=200 nm)
], dtype=torch.float64)

# 磁偶极矩(方向和大小随意示例)
# 假设 3 个偶极子的磁矩各不相同
dipole_moments = torch.tensor([
    [1e-13, 0.0,   0.0  ],  # 单位 A·m^2，举例
    [0.0,    1e-13,0.0  ],
    [0.0,    0.0,   1e-13],
], dtype=torch.float64)


# -----------------------------
# 2. 生成观测平面上的坐标网格
# -----------------------------
# 假设我们想在 x,y ∈ [-300nm, 300nm] 的区域采样 51×51 个点
nx = 51
ny = 51
x_min, x_max = -3e-7, 3e-7  # -300 nm ~ 300 nm
y_min, y_max = -3e-7, 3e-7

xs = torch.linspace(x_min, x_max, nx, dtype=torch.float64)
ys = torch.linspace(y_min, y_max, ny, dtype=torch.float64)

# 利用 meshgrid 生成所有 (x,y) 点，再拼到一起
XX, YY = torch.meshgrid(xs, ys, indexing='ij')  # shape (nx, ny)
ZZ = torch.full_like(XX, z_obs)                 # z = 100 nm

# 把观测点展开为 (M,3) 形状, M=nx*ny
points = torch.stack([XX, YY, ZZ], dim=-1).view(-1, 3)  # shape (M, 3)


# -----------------------------
# 3. 定义计算磁偶极子场的函数
# -----------------------------
def dipoles_field_at_points(dip_pos, dip_mom, query_points):
    """计算磁偶极子在查询点处产生的磁场 (NumPy版本)
    
    Args:
        dip_pos: 磁偶极子位置，形状为(n_dipoles, 3)
        dip_mom: 磁偶极子矩，形状为(n_dipoles, 3)
        query_points: 查询点位置，形状为(n_points, 3)
        
    Returns:
        查询点处的磁场，形状为(n_points, 3)
    """
    # 确保所有输入都是NumPy数组
    if isinstance(dip_pos, torch.Tensor):
        dip_pos = dip_pos.detach().cpu().numpy()
    if isinstance(dip_mom, torch.Tensor):
        dip_mom = dip_mom.detach().cpu().numpy()
    if isinstance(query_points, torch.Tensor):
        query_points = query_points.detach().cpu().numpy()
    
    # 常数
    mu0_div_4pi = 1e-7  # μ0/4π (T·m/A)
    
    n_points = query_points.shape[0]
    n_dipoles = dip_pos.shape[0]
    
    # 初始化磁场
    B_field = np.zeros((n_points, 3), dtype=np.float32)
    
    # 计算每个偶极子产生的磁场
    for i in range(n_dipoles):
        # 计算r向量（从偶极子到查询点）
        r_vec = query_points - dip_pos[i]
        
        # 计算r的模
        r_norm = np.linalg.norm(r_vec, axis=1)
        
        # 避免除以零
        mask = r_norm > 1e-10
        
        # 单位r向量
        r_hat = np.zeros_like(r_vec)
        r_hat[mask] = r_vec[mask] / r_norm[mask, np.newaxis]
        
        # 计算磁场 B = (μ0/4π) * (3(m·r̂)r̂ - m) / r³
        m_dot_r = np.sum(dip_mom[i] * r_hat, axis=1)
        
        # 计算每个查询点的磁场
        for j in range(n_points):
            if r_norm[j] > 1e-10:  # 避免除以零
                B_field[j] += mu0_div_4pi * (3 * m_dot_r[j] * r_hat[j] - dip_mom[i]) / (r_norm[j]**3)
    
    return B_field

def dipoles_field_at_points_vectorized(dip_pos, dip_mom, query_points):
    """计算磁偶极子在查询点处产生的磁场 (向量化NumPy版本)
    
    Args:
        dip_pos: 磁偶极子位置，形状为(n_dipoles, 3)
        dip_mom: 磁偶极子矩，形状为(n_dipoles, 3)
        query_points: 查询点位置，形状为(n_points, 3)
        
    Returns:
        查询点处的磁场，形状为(n_points, 3)
    """
    # 确保所有输入都是NumPy数组
    if isinstance(dip_pos, torch.Tensor):
        dip_pos = dip_pos.detach().cpu().numpy()
    if isinstance(dip_mom, torch.Tensor):
        dip_mom = dip_mom.detach().cpu().numpy()
    if isinstance(query_points, torch.Tensor):
        query_points = query_points.detach().cpu().numpy()
    
    # 常数
    mu0_div_4pi = 1e-7  # μ0/4π (T·m/A)
    
    n_points = query_points.shape[0]
    n_dipoles = dip_pos.shape[0]
    
    # 初始化磁场
    B_field = np.zeros((n_points, 3), dtype=np.float32)
    
    # 向量化计算
    for i in range(n_dipoles):
        # 计算r向量（从偶极子到查询点）
        r_vec = query_points - dip_pos[i]
        
        # 计算r的模
        r_norm = np.linalg.norm(r_vec, axis=1)
        
        # 避免除以零
        mask = r_norm > 1e-10
        
        if np.any(mask):  # 只在有效点上计算
            # 单位r向量
            r_hat = np.zeros_like(r_vec)
            r_hat[mask] = r_vec[mask] / r_norm[mask, np.newaxis]
            
            # 计算磁场 B = (μ0/4π) * (3(m·r̂)r̂ - m) / r³
            m_dot_r = np.sum(dip_mom[i] * r_hat, axis=1)
            
            # 向量化计算所有点的磁场贡献
            B_contribution = np.zeros_like(B_field)
            B_contribution[mask] = (mu0_div_4pi * (3 * m_dot_r[mask, np.newaxis] * 
                                   r_hat[mask] - dip_mom[i]) / 
                                   (r_norm[mask, np.newaxis]**3))
            
            B_field += B_contribution
    
    return B_field


# -----------------------------
# 4. 计算结果并 reshape
# -----------------------------
# 使用向量化版本计算磁场
B = dipoles_field_at_points_vectorized(dipole_positions, dipole_moments, points)

# 确保 B 是 NumPy 数组
if isinstance(B, torch.Tensor):
    B = B.detach().cpu().numpy()

# 然后使用 reshape
B = B.reshape(nx, ny, 3)

# 分离 x, y, z 分量
Bx = B[:, :, 0]
By = B[:, :, 1]
Bz = B[:, :, 2]

# 计算磁场强度
B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)

# -----------------------------
# 5. 后续可视化或分析
# -----------------------------
# 可视化磁场分量
plt.figure(figsize=(15, 10))

# Bx 分量
plt.subplot(2, 2, 1)
plt.imshow(Bx, cmap='gray')  # 使用灰度图
plt.colorbar(label='Bx [T]')
plt.title('Bx Component')

# By 分量
plt.subplot(2, 2, 2)
plt.imshow(By, cmap='gray')  # 使用灰度图
plt.colorbar(label='By [T]')
plt.title('By Component')

# Bz 分量
plt.subplot(2, 2, 3)
plt.imshow(Bz, cmap='gray')  # 使用灰度图
plt.colorbar(label='Bz [T]')
plt.title('Bz Component')

# 磁场强度
plt.subplot(2, 2, 4)
plt.imshow(B_magnitude, cmap='gray')  # 使用灰度图
plt.colorbar(label='|B| [T]')
plt.title('Magnetic Field Magnitude')

plt.tight_layout()
plt.savefig('magnetic_field_components.png')
plt.close()

# 保存灰度图版本的磁场数据为图像文件
# 归一化并转换为灰度图
Bx_norm = ((Bx - np.min(Bx)) / (np.max(Bx) - np.min(Bx)) * 255).astype(np.uint8)
By_norm = ((By - np.min(By)) / (np.max(By) - np.min(By)) * 255).astype(np.uint8)
Bz_norm = ((Bz - np.min(Bz)) / (np.max(Bz) - np.min(Bz)) * 255).astype(np.uint8)
B_mag_norm = ((B_magnitude - np.min(B_magnitude)) / 
             (np.max(B_magnitude) - np.min(B_magnitude)) * 255).astype(np.uint8)

# 保存为PNG文件
Image.fromarray(Bx_norm).save('Bx_gray.png')
Image.fromarray(By_norm).save('By_gray.png')
Image.fromarray(Bz_norm).save('Bz_gray.png')
Image.fromarray(B_mag_norm).save('B_magnitude_gray.png')

def generate_random_magnetic_data(num_samples=10, num_dipoles=50, image_height=540, image_width=720, 
                                  field_of_view=10.0, dipole_strength_range=(1e-9, 1e-8),
                                  noise_level=0.05, save_dir='magnetic_simulation_data'):
    """生成随机磁荷分布的模拟数据
    
    Args:
        num_samples: 要生成的样本数量
        num_dipoles: 每个样本中的磁偶极子数量
        image_height: 图像高度（像素）
        image_width: 图像宽度（像素）
        field_of_view: 视场大小（毫米）
        dipole_strength_range: 磁偶极子强度范围（Am²）
        noise_level: 添加的噪声水平（相对于信号最大值的比例）
        save_dir: 保存数据的目录
        
    Returns:
        保存的文件路径列表
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建像素网格
    x = np.linspace(-field_of_view/2, field_of_view/2, image_width)
    y = np.linspace(-field_of_view/2, field_of_view/2, image_height)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx) + 0.5  # 假设传感器在z=0.5mm平面
    
    # 将网格点转换为查询点列表
    query_points = np.stack([xx.flatten(), yy.flatten(), z.flatten()], axis=1)
    
    # 生成频率轴（模拟ODMR频谱）
    freq_start = 2700  # MHz
    freq_end = 3000    # MHz
    num_freq_points = 300
    frequencies = np.linspace(freq_start, freq_end, num_freq_points)
    
    # NV中心参数
    nv_axes = np.array([
        [1, 1, 1],    # NV1
        [1, -1, -1],  # NV2
        [-1, 1, -1],  # NV3
        [-1, -1, 1]   # NV4
    ]) / np.sqrt(3)
    
    # 中心频率和灵敏度
    center_freq = 2870  # MHz
    sensitivity = 2.8   # MHz/G
    
    saved_files = []
    
    for sample_idx in tqdm(range(num_samples), desc="生成模拟数据"):
        # 随机生成磁偶极子位置（在视场范围内，但z位置在传感器下方）
        dip_pos = np.random.uniform(
            low=[-field_of_view/2, -field_of_view/2, -field_of_view/2],
            high=[field_of_view/2, field_of_view/2, 0],
            size=(num_dipoles, 3)
        )
        
        # 随机生成磁偶极子矩（强度在给定范围内，方向随机）
        dipole_strengths = np.random.uniform(
            low=dipole_strength_range[0],
            high=dipole_strength_range[1],
            size=num_dipoles
        )
        
        # 随机方向
        dipole_directions = np.random.randn(num_dipoles, 3)
        dipole_directions /= np.linalg.norm(dipole_directions, axis=1)[:, np.newaxis]
        
        # 计算磁偶极子矩
        dip_mom = dipole_strengths[:, np.newaxis] * dipole_directions
        
        # 计算查询点处的磁场
        B_field = dipoles_field_at_points(dip_pos, dip_mom, query_points)
        
        # 确保 B_field 是 NumPy 数组
        if isinstance(B_field, torch.Tensor):
            B_field = B_field.detach().cpu().numpy()
        
        # 重塑为图像尺寸
        B_field = B_field.reshape(image_height, image_width, 3)
        
        # 分离 x, y, z 分量
        Bx = B_field[:, :, 0]
        By = B_field[:, :, 1]
        Bz = B_field[:, :, 2]
        
        # 计算磁场强度
        B_magnitude = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        # 添加一些空间平滑以模拟实际测量
        Bx = gaussian_filter(Bx, sigma=1.0)
        By = gaussian_filter(By, sigma=1.0)
        Bz = gaussian_filter(Bz, sigma=1.0)
        B_magnitude = gaussian_filter(B_magnitude, sigma=1.0)
        
        # 生成ODMR频谱数据 - 确保数据适合灰度图显示
        odmr_data = np.zeros((num_freq_points, image_height, image_width))
        
        # 背景信号强度（均匀）
        background = np.ones((image_height, image_width)) * np.random.uniform(30000, 35000)
        
        # 添加一些空间变化，使灰度图更有层次感
        background += gaussian_filter(np.random.normal(0, 1000, (image_height, image_width)), sigma=50)
        
        # 峰强度（随机变化）
        peak_amplitude = np.random.uniform(500, 700)
        
        # 峰宽度（随机变化）
        peak_width = np.random.uniform(8, 12)
        
        # 计算每个NV轴向的磁场投影
        for i, axis in enumerate(nv_axes):
            # 计算磁场在NV轴上的投影
            B_projection = Bx * axis[0] + By * axis[1] + Bz * axis[2]
            
            # 计算频率偏移
            freq_shift = B_projection * sensitivity  # MHz
            
            # 生成两个峰（对应正负磁场方向）
            peak1_center = center_freq + freq_shift
            peak2_center = center_freq - freq_shift
            
            # 为每个频率点和每个像素生成洛伦兹峰
            for freq_idx, freq in enumerate(frequencies):
                # 第一个峰（正方向）
                peak1 = peak_amplitude / (((freq - peak1_center)/peak_width)**2 + 1)
                
                # 第二个峰（负方向）
                peak2 = peak_amplitude / (((freq - peak2_center)/peak_width)**2 + 1)
                
                # 将峰添加到ODMR数据中（从背景中减去，因为是向下的峰）
                odmr_data[freq_idx] = odmr_data[freq_idx] - peak1 - peak2
            
        # 添加背景
        for freq_idx in range(num_freq_points):
            odmr_data[freq_idx] += background
        
        # 添加噪声
        noise = np.random.normal(0, noise_level * peak_amplitude, odmr_data.shape)
        odmr_data += noise
        
        # 确保数据为正值
        odmr_data = np.maximum(odmr_data, 0)
        
        # 保存数据
        filename = os.path.join(save_dir, f'magnetic_sample_{sample_idx:03d}.h5')
        with h5py.File(filename, 'w') as f:
            # 保存ODMR数据
            f.create_dataset('odmr_data', data=odmr_data)
            
            # 保存频率轴
            f.create_dataset('frequencies', data=frequencies)
            
            # 保存真实磁场（用于验证）
            f.create_dataset('Bx', data=Bx)
            f.create_dataset('By', data=By)
            f.create_dataset('Bz', data=Bz)
            f.create_dataset('B_magnitude', data=B_magnitude)
            
            # 保存灰度图版本的磁场数据
            f.create_dataset('Bx_gray', data=((Bx - np.min(Bx)) / (np.max(Bx) - np.min(Bx)) * 255).astype(np.uint8))
            f.create_dataset('By_gray', data=((By - np.min(By)) / (np.max(By) - np.min(By)) * 255).astype(np.uint8))
            f.create_dataset('Bz_gray', data=((Bz - np.min(Bz)) / (np.max(Bz) - np.min(Bz)) * 255).astype(np.uint8))
            f.create_dataset('B_magnitude_gray', data=((B_magnitude - np.min(B_magnitude)) / 
                                                     (np.max(B_magnitude) - np.min(B_magnitude)) * 255).astype(np.uint8))
            
            # 保存磁偶极子信息（用于验证）
            f.create_dataset('dipole_positions', data=dip_pos)
            f.create_dataset('dipole_moments', data=dip_mom)
            
            # 保存参数
            f.attrs['background'] = background.mean()
            f.attrs['peak_amplitude'] = peak_amplitude
            f.attrs['peak_width'] = peak_width
            f.attrs['center_frequency'] = center_freq
            f.attrs['sensitivity'] = sensitivity
        
        saved_files.append(filename)
        
        # 可视化第一个样本
        if sample_idx == 0:
            # 绘制磁场灰度图
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(Bx * 1e4, cmap='gray')  # 使用灰度图
            plt.colorbar(label='Bx (G)')
            plt.title('Bx分量')
            
            plt.subplot(2, 3, 2)
            plt.imshow(By * 1e4, cmap='gray')
            plt.colorbar(label='By (G)')
            plt.title('By分量')
            
            plt.subplot(2, 3, 3)
            plt.imshow(Bz * 1e4, cmap='gray')
            plt.colorbar(label='Bz (G)')
            plt.title('Bz分量')
            
            plt.subplot(2, 3, 4)
            plt.imshow(B_magnitude * 1e4, cmap='gray')
            plt.colorbar(label='|B| (G)')
            plt.title('磁场强度')
            
            # 绘制磁偶极子位置
            plt.subplot(2, 3, 5)
            plt.scatter(dip_pos[:, 0], dip_pos[:, 1], c='black', s=10)
            plt.xlim(-field_of_view/2, field_of_view/2)
            plt.ylim(-field_of_view/2, field_of_view/2)
            plt.title(f'磁偶极子位置 (n={num_dipoles})')
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            plt.gca().set_facecolor('white')  # 白色背景
            
            # 绘制一个像素位置的ODMR频谱
            plt.subplot(2, 3, 6)
            center_pixel = odmr_data[:, image_height//2, image_width//2]
            plt.plot(frequencies, center_pixel, 'k-')  # 黑色线条
            plt.xlabel('频率 (MHz)')
            plt.ylabel('强度')
            plt.title('中心像素ODMR频谱')
            plt.gca().set_facecolor('white')  # 白色背景
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'sample_visualization.png'))
            plt.close()
            
            # 保存灰度图版本的磁场数据为图像文件
            # 归一化并转换为灰度图
            Bx_norm = ((Bx - np.min(Bx)) / (np.max(Bx) - np.min(Bx)) * 255).astype(np.uint8)
            By_norm = ((By - np.min(By)) / (np.max(By) - np.min(By)) * 255).astype(np.uint8)
            Bz_norm = ((Bz - np.min(Bz)) / (np.max(Bz) - np.min(Bz)) * 255).astype(np.uint8)
            B_mag_norm = ((B_magnitude - np.min(B_magnitude)) / 
                         (np.max(B_magnitude) - np.min(B_magnitude)) * 255).astype(np.uint8)
            
            # 保存为PNG文件
            Image.fromarray(Bx_norm).save(os.path.join(save_dir, 'Bx_gray.png'))
            Image.fromarray(By_norm).save(os.path.join(save_dir, 'By_gray.png'))
            Image.fromarray(Bz_norm).save(os.path.join(save_dir, 'Bz_gray.png'))
            Image.fromarray(B_mag_norm).save(os.path.join(save_dir, 'B_magnitude_gray.png'))
    
    print(f"已生成{num_samples}个模拟样本，保存在{save_dir}目录中")
    return saved_files

def generate_dipole_ground_truth(save_dir='magnetic_simulation_data', field_of_view=10.0, 
                                image_height=540, image_width=720, visualize=True):
    """从已保存的模拟数据中提取磁偶极子位置的地面真值并可视化
    
    Args:
        save_dir: 保存数据的目录
        field_of_view: 视场大小（毫米）
        image_height: 图像高度（像素）
        image_width: 图像宽度（像素）
        visualize: 是否生成可视化图像
        
    Returns:
        包含所有样本磁偶极子位置的字典
    """
    import glob
    import os
    
    # 查找所有h5文件
    h5_files = glob.glob(os.path.join(save_dir, '*.h5'))
    
    if not h5_files:
        print(f"在{save_dir}目录中未找到h5文件")
        return {}
    
    # 创建保存地面真值的目录
    gt_dir = os.path.join(save_dir, 'ground_truth')
    os.makedirs(gt_dir, exist_ok=True)
    
    # 存储所有样本的磁偶极子位置
    all_dipoles = {}
    
    for file_path in tqdm(h5_files, desc="提取地面真值"):
        sample_id = os.path.basename(file_path).split('.')[0]
        
        with h5py.File(file_path, 'r') as f:
            # 提取磁偶极子位置和矩
            dipole_positions = f['dipole_positions'][:]
            dipole_moments = f['dipole_moments'][:]
            
            # 计算磁偶极子强度
            dipole_strengths = np.linalg.norm(dipole_moments, axis=1)
            
            # 保存到字典
            all_dipoles[sample_id] = {
                'positions': dipole_positions,
                'moments': dipole_moments,
                'strengths': dipole_strengths
            }
            
            # 可视化
            if visualize:
                plt.figure(figsize=(12, 10))
                
                # 绘制磁偶极子位置的俯视图
                plt.subplot(2, 2, 1)
                scatter = plt.scatter(dipole_positions[:, 0], dipole_positions[:, 1], 
                          c=dipole_strengths, cmap='viridis', s=30, alpha=0.8)
                plt.colorbar(scatter, label='磁矩强度 (Am^2)')
                plt.xlim(-field_of_view/2, field_of_view/2)
                plt.ylim(-field_of_view/2, field_of_view/2)
                plt.title(f'磁偶极子位置俯视图 ({len(dipole_positions)}个)')
                plt.xlabel('x (mm)')
                plt.ylabel('y (mm)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制磁偶极子位置的侧视图
                plt.subplot(2, 2, 2)
                scatter = plt.scatter(dipole_positions[:, 0], dipole_positions[:, 2], 
                          c=dipole_strengths, cmap='viridis', s=30, alpha=0.8)
                plt.colorbar(scatter, label='磁矩强度 (Am^2)')
                plt.xlim(-field_of_view/2, field_of_view/2)
                plt.ylim(-field_of_view/2, 0)  # z轴通常是负值（在传感器下方）
                plt.title('磁偶极子位置侧视图 (x-z平面)')
                plt.xlabel('x (mm)')
                plt.ylabel('z (mm)')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # 绘制磁偶极子位置的3D视图
                ax = plt.subplot(2, 2, 3, projection='3d')
                scatter = ax.scatter(dipole_positions[:, 0], dipole_positions[:, 1], dipole_positions[:, 2],
                          c=dipole_strengths, cmap='viridis', s=30, alpha=0.8)
                plt.colorbar(scatter, label='磁矩强度 (Am^2)')
                ax.set_xlim(-field_of_view/2, field_of_view/2)
                ax.set_ylim(-field_of_view/2, field_of_view/2)
                ax.set_zlim(-field_of_view/2, 0)
                ax.set_title('磁偶极子位置3D视图')
                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')
                ax.set_zlabel('z (mm)')
                
                # 绘制磁矩方向
                ax = plt.subplot(2, 2, 4, projection='3d')
                # 归一化磁矩方向
                moment_directions = dipole_moments / np.linalg.norm(dipole_moments, axis=1)[:, np.newaxis]
                # 绘制位置点
                ax.scatter(dipole_positions[:, 0], dipole_positions[:, 1], dipole_positions[:, 2],
                          c=dipole_strengths, cmap='viridis', s=30, alpha=0.8)
                # 绘制方向箭头（只显示部分点以避免过于拥挤）
                max_arrows = min(20, len(dipole_positions))
                indices = np.linspace(0, len(dipole_positions)-1, max_arrows, dtype=int)
                for i in indices:
                    ax.quiver(dipole_positions[i, 0], dipole_positions[i, 1], dipole_positions[i, 2],
                             moment_directions[i, 0], moment_directions[i, 1], moment_directions[i, 2],
                             length=field_of_view/10, color='red', alpha=0.6)
                ax.set_xlim(-field_of_view/2, field_of_view/2)
                ax.set_ylim(-field_of_view/2, field_of_view/2)
                ax.set_zlim(-field_of_view/2, 0)
                ax.set_title('磁偶极子方向')
                ax.set_xlabel('x (mm)')
                ax.set_ylabel('y (mm)')
                ax.set_zlabel('z (mm)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(gt_dir, f'{sample_id}_ground_truth.png'))
                plt.close()
                
                # 保存磁偶极子位置的热力图
                plt.figure(figsize=(10, 8))
                
                # 创建2D直方图
                hist, xedges, yedges = np.histogram2d(
                    dipole_positions[:, 0], dipole_positions[:, 1],
                    bins=[image_width//10, image_height//10],
                    range=[[-field_of_view/2, field_of_view/2], [-field_of_view/2, field_of_view/2]]
                )
                
                # 绘制热力图
                plt.imshow(hist.T, origin='lower', extent=[-field_of_view/2, field_of_view/2, -field_of_view/2, field_of_view/2],
                          cmap='hot', interpolation='gaussian')
                plt.colorbar(label='磁偶极子数量')
                plt.title('磁偶极子位置密度热力图')
                plt.xlabel('x (mm)')
                plt.ylabel('y (mm)')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(gt_dir, f'{sample_id}_density_heatmap.png'))
                plt.close()
            
            # 保存地面真值数据为CSV文件
            csv_path = os.path.join(gt_dir, f'{sample_id}_dipoles.csv')
            with open(csv_path, 'w', encoding='utf-8') as csv_file:
                csv_file.write('x,y,z,moment_x,moment_y,moment_z,strength\n')
                for i in range(len(dipole_positions)):
                    csv_file.write(f"{dipole_positions[i,0]},{dipole_positions[i,1]},{dipole_positions[i,2]},")
                    csv_file.write(f"{dipole_moments[i,0]},{dipole_moments[i,1]},{dipole_moments[i,2]},")
                    csv_file.write(f"{dipole_strengths[i]}\n")
    
    # 创建汇总报告
    with open(os.path.join(gt_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"磁偶极子地面真值汇总报告\n")
        f.write(f"=======================\n\n")
        f.write(f"总样本数: {len(all_dipoles)}\n\n")
        
        for sample_id, data in all_dipoles.items():
            f.write(f"样本 {sample_id}:\n")
            f.write(f"  磁偶极子数量: {len(data['positions'])}\n")
            f.write(f"  磁矩强度范围: {data['strengths'].min():.2e} - {data['strengths'].max():.2e} Am^2\n")
            f.write(f"  磁矩强度均值: {data['strengths'].mean():.2e} Am^2\n")
            f.write(f"  磁偶极子深度范围: {data['positions'][:,2].min():.2f} - {data['positions'][:,2].max():.2f} mm\n\n")
    
    print(f"已提取{len(all_dipoles)}个样本的地面真值，保存在{gt_dir}目录中")
    return all_dipoles

def main():
    # 生成10组模拟数据，每组约50个磁荷，图像尺寸为540*720
    saved_files = generate_random_magnetic_data(
        num_samples=10,
        num_dipoles=50,
        image_height=540,
        image_width=720,
        field_of_view=10.0,  # 10mm视场
        dipole_strength_range=(1e-9, 1e-8),  # 磁偶极子强度范围
        noise_level=0.05,  # 5%的噪声水平
        save_dir='magnetic_simulation_data'
    )
    
    # 打印生成的文件
    print("生成的文件:")
    for file in saved_files:
        print(f" - {file}")
    
    # 加载并测试生成的数据
    print("\n测试加载生成的数据:")
    with h5py.File(saved_files[0], 'r') as f:
        odmr_data = f['odmr_data'][:]
        frequencies = f['frequencies'][:]
        
        print(f"ODMR数据形状: {odmr_data.shape}")
        print(f"频率范围: {frequencies[0]:.1f} - {frequencies[-1]:.1f} MHz")
        print(f"背景强度: {f.attrs['background']:.1f}")
        print(f"峰振幅: {f.attrs['peak_amplitude']:.1f}")
        print(f"峰宽度: {f.attrs['peak_width']:.1f} MHz")
    
    # 提取并可视化地面真值
    all_dipoles = generate_dipole_ground_truth(
        save_dir='magnetic_simulation_data',
        field_of_view=10.0,
        image_height=540,
        image_width=720
    )

if __name__ == "__main__":
    main()
