"""
PSF模拟器 - 用于生成模拟的点扩散函数和训练数据
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import math


class GaussianPSF:
    """高斯点扩散函数模拟器"""
    
    def __init__(self, 
                 sigma_x: float = 1.0,
                 sigma_y: float = 1.0,
                 sigma_z: float = 2.0,
                 pixel_size: float = 100.0,  # 纳米
                 wavelength: float = 680.0,  # 纳米
                 numerical_aperture: float = 1.4):
        """
        初始化高斯PSF模拟器
        
        Args:
            sigma_x: x方向的标准差
            sigma_y: y方向的标准差
            sigma_z: z方向的标准差
            pixel_size: 像素大小（纳米）
            wavelength: 波长（纳米）
            numerical_aperture: 数值孔径
        """
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.na = numerical_aperture
        
        # 计算理论PSF标准差
        self.sigma_theo = 0.21 * wavelength / numerical_aperture / pixel_size

    def generate_psf(self, 
                    size: Tuple[int, int] = (11, 11),
                    z: float = 0.0) -> torch.Tensor:
        """
        生成高斯PSF
        
        Args:
            size: PSF的大小 (height, width)
            z: z位置（纳米）
            
        Returns:
            PSF张量 [height, width]
        """
        height, width = size
        
        # 计算z依赖的sigma
        sigma_factor = 1.0 + (z / 1000.0) ** 2  # z依赖因子
        sigma_x = self.sigma_x * sigma_factor
        sigma_y = self.sigma_y * sigma_factor
        
        # 确定设备
        device = z.device if isinstance(z, torch.Tensor) else torch.device('cpu')
        
        # 创建网格 - 添加indexing参数
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij'  # 显式指定索引方式
        )
        
        # 中心坐标
        x0 = width / 2
        y0 = height / 2
        
        # 计算高斯分布
        gaussian = torch.exp(-(
            ((x - x0) ** 2) / (2 * sigma_x ** 2) + 
            ((y - y0) ** 2) / (2 * sigma_y ** 2)
        ))
        
        # 归一化
        gaussian = gaussian / gaussian.sum()
        
        return gaussian


class EmitterSimulator:
    """发射体模拟器"""
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (128, 128),
                 psf_model: Optional[GaussianPSF] = None,
                 density_range: Tuple[float, float] = (0.1, 5.0),
                 intensity_range: Tuple[float, float] = (500, 5000),
                 bg_range: Tuple[float, float] = (50, 200),
                 lifetime_frames: Tuple[int, int] = (1, 10)):
        """
        初始化发射体模拟器
        
        Args:
            img_size: 图像大小 (height, width)
            psf_model: PSF模型
            density_range: 每平方微米的发射体密度范围
            intensity_range: 发射体强度范围
            bg_range: 背景噪声范围
            lifetime_frames: 发射体寿命范围（帧数）
        """
        self.img_size = img_size
        self.psf_model = psf_model or GaussianPSF()
        self.density_range = density_range
        self.intensity_range = intensity_range
        self.bg_range = bg_range
        self.lifetime_frames = lifetime_frames
        
        # PSF大小
        self.psf_size = (11, 11)

    def generate_random_emitters(self, 
                                 num_frames: int = 5,
                                 device: torch.device = torch.device('cpu')) -> Dict[str, torch.Tensor]:
        """
        生成随机发射体
        
        Args:
            num_frames: 帧数
            device: 设备
            
        Returns:
            包含发射体参数的字典
        """
        height, width = self.img_size
        
        # 确定发射体密度
        density = np.random.uniform(*self.density_range)
        
        # 计算发射体数量（考虑图像面积）
        area_um2 = (height * width) * (self.psf_model.pixel_size / 1000) ** 2
        num_emitters = int(density * area_um2)
        
        # 随机生成发射体位置
        x_pos = torch.rand(num_emitters, device=device) * width
        y_pos = torch.rand(num_emitters, device=device) * height
        z_pos = torch.randn(num_emitters, device=device) * 500  # z位置（纳米）
        
        # 随机生成发射体强度
        intensity = torch.rand(num_emitters, device=device) * (
            self.intensity_range[1] - self.intensity_range[0]
        ) + self.intensity_range[0]
        
        # 随机生成发射体寿命
        lifetime = torch.randint(
            self.lifetime_frames[0],
            self.lifetime_frames[1] + 1,
            (num_emitters,),
            device=device
        )
        
        # 随机生成发射体出现帧
        start_frame = torch.randint(
            0,
            num_frames - self.lifetime_frames[0] + 1,
            (num_emitters,),
            device=device
        )
        
        return {
            'x_pos': x_pos,
            'y_pos': y_pos,
            'z_pos': z_pos,
            'intensity': intensity,
            'lifetime': lifetime,
            'start_frame': start_frame
        }

    def generate_frames(self, 
                        emitters: Dict[str, torch.Tensor],
                        num_frames: int = 5,
                        add_noise: bool = True,
                        device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        生成包含发射体的帧序列
        
        Args:
            emitters: 发射体参数
            num_frames: 帧数
            add_noise: 是否添加噪声
            device: 设备
            
        Returns:
            帧序列张量 [num_frames, height, width]
        """
        height, width = self.img_size
        
        # 创建空白帧序列
        frames = torch.zeros((num_frames, height, width), device=device)
        
        # 随机背景噪声
        if add_noise:
            bg_level = torch.rand(1, device=device) * (
                self.bg_range[1] - self.bg_range[0]
            ) + self.bg_range[0]
            frames += bg_level
        
        # 添加发射体
        for i in range(len(emitters['x_pos'])):
            x = emitters['x_pos'][i]
            y = emitters['y_pos'][i]
            z = emitters['z_pos'][i]
            intensity = emitters['intensity'][i]
            lifetime = emitters['lifetime'][i]
            start_frame = emitters['start_frame'][i]
            
            # 发射体活跃的帧
            active_frames = range(start_frame, min(start_frame + lifetime, num_frames))
            
            # 获取PSF
            psf = self.psf_model.generate_psf(self.psf_size, z)
            psf = psf.to(device)
            
            # 将PSF添加到帧中
            for frame_idx in active_frames:
                # 计算PSF中心位置
                x_center = int(x.item())
                y_center = int(y.item())
                
                # 计算PSF区域
                psf_h, psf_w = self.psf_size
                x_start = max(0, x_center - psf_w // 2)
                y_start = max(0, y_center - psf_h // 2)
                x_end = min(width, x_center + psf_w // 2 + 1)
                y_end = min(height, y_center + psf_h // 2 + 1)
                
                # 计算PSF区域在PSF中的位置
                psf_x_start = max(0, psf_w // 2 - x_center)
                psf_y_start = max(0, psf_h // 2 - y_center)
                psf_x_end = psf_w - max(0, x_center + psf_w // 2 + 1 - width)
                psf_y_end = psf_h - max(0, y_center + psf_h // 2 + 1 - height)
                
                # 添加PSF到帧中
                frames[frame_idx, y_start:y_end, x_start:x_end] += (
                    psf[psf_y_start:psf_y_end, psf_x_start:psf_x_end] * intensity
                )
        
        # 添加泊松噪声
        if add_noise:
            frames = torch.poisson(frames)
        
        return frames

    def generate_training_data(self, 
                              num_samples: int = 100,
                              num_frames: int = 5,
                              device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        生成训练数据
        
        Args:
            num_samples: 样本数量
            num_frames: 每个样本的帧数
            device: 设备
            
        Returns:
            (输入帧张量 [num_samples, num_frames, height, width], 标签字典)
        """
        height, width = self.img_size
        
        # 创建输入和标签
        inputs = torch.zeros((num_samples, num_frames, height, width), device=device)
        labels = {
            'detection_map': torch.zeros((num_samples, 1, height, width), device=device),
            'x_coord': torch.zeros((num_samples, 1, height, width), device=device),
            'y_coord': torch.zeros((num_samples, 1, height, width), device=device),
            'z_coord': torch.zeros((num_samples, 1, height, width), device=device),
            'intensity': torch.zeros((num_samples, 1, height, width), device=device),
            'background': torch.zeros((num_samples, 1, height, width), device=device),
            'uncertainty': torch.zeros((num_samples, 1, height, width), device=device)
        }
        
        # 生成样本
        for i in range(num_samples):
            # 生成发射体
            emitters = self.generate_random_emitters(num_frames, device)
            
            # 生成帧序列
            frames = self.generate_frames(emitters, num_frames, True, device)
            inputs[i] = frames
            
            # 生成标签
            # 中心帧索引
            center_frame = num_frames // 2
            
            # 计算在中心帧活跃的发射体
            active_mask = (
                (emitters['start_frame'] <= center_frame) & 
                (emitters['start_frame'] + emitters['lifetime'] > center_frame)
            )
            
            # 提取活跃发射体
            active_x = emitters['x_pos'][active_mask]
            active_y = emitters['y_pos'][active_mask]
            active_z = emitters['z_pos'][active_mask]
            active_intensity = emitters['intensity'][active_mask]
            
            # 创建检测图
            for j in range(len(active_x)):
                x = int(active_x[j].item())
                y = int(active_y[j].item())
                
                if 0 <= x < width and 0 <= y < height:
                    # 设置检测点
                    labels['detection_map'][i, 0, y, x] = 1.0
                    
                    # 设置坐标和参数
                    labels['x_coord'][i, 0, y, x] = active_x[j] - x  # 亚像素位置
                    labels['y_coord'][i, 0, y, x] = active_y[j] - y  # 亚像素位置
                    labels['z_coord'][i, 0, y, x] = active_z[j]
                    labels['intensity'][i, 0, y, x] = active_intensity[j]
                    
                    # 设置不确定性（简化为常数）
                    labels['uncertainty'][i, 0, y, x] = 0.1
            
            # 计算背景
            bg_mask = labels['detection_map'][i, 0] == 0
            bg_value = inputs[i, center_frame][bg_mask].mean()
            labels['background'][i, 0] = bg_value
        
        return inputs, labels 