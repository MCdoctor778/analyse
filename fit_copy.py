import os
# 设置环境变量以解决OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import re
from scipy.signal import savgol_filter
import pywt
from scipy.fft import fft, ifft
import time

# 配置参数
IMAGE_HEIGHT = 540
IMAGE_WIDTH = 720
FREQ_START = 2720  # MHz
FREQ_END = 3018    # MHz
WINDOW_SIZE = 30    # 空间滑动窗口大小 - 从8修改为30
STEP = 1            # 滑动步长
NUM_WORKERS = 24     # 并行线程数 - 从8修改为24，为32核心CPU的75%
GPU_BATCH = 2048    # GPU批处理大小

# 自动检测GPU并设置设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 如果有GPU，获取GPU信息
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}, 内存: {gpu_mem:.2f} GB")
    
    # 根据GPU内存自动调整批处理大小
    if gpu_mem > 8:
        GPU_BATCH = 4096
    elif gpu_mem > 4:
        GPU_BATCH = 2048
    else:
        GPU_BATCH = 1024
    print(f"自动设置GPU批处理大小: {GPU_BATCH}")

# 计算滑动窗口后的尺寸
PROCESSED_HEIGHT = IMAGE_HEIGHT - WINDOW_SIZE + 1  # 滑动窗口后的高度
PROCESSED_WIDTH = IMAGE_WIDTH - WINDOW_SIZE + 1    # 滑动窗口后的宽度

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class DataLoader:
    @staticmethod
    def natural_sort_key(s):
        """自然排序键函数"""
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]

    @classmethod
    def load_folder(cls, folder_path: str) -> np.ndarray:
        """加载整个文件夹的RAW数据并返回三维数组"""
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.RAW')],
                      key=cls.natural_sort_key)
        
        if len(files) != FREQ_END - FREQ_START + 1:
            raise ValueError(f"文件数量不匹配！预期{FREQ_END-FREQ_START+1}个，找到{len(files)}个")
        
        # 预分配内存
        data = np.empty((len(files), IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i, filename in enumerate(files):
                file_path = os.path.join(folder_path, filename)
                futures.append((i, executor.submit(DataLoader._read_raw, file_path)))
            
            for i, future in tqdm(futures, desc=f"加载 {folder_path}"):
                data[i] = future.result()
        
        return data

    @staticmethod
    def _read_raw(file_path: str) -> np.ndarray:
        """读取单个RAW文件"""
        return np.fromfile(file_path, dtype=np.uint16).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

class SpatialProcessor:
    @staticmethod
    def sliding_window(data: np.ndarray) -> np.ndarray:
        """应用滑动窗口进行空间平均 - GPU加速版本"""
        print(f"滑动窗口前数据形状: {data.shape}")
        
        # 将数据转移到GPU，并转换为浮点类型
        start_time = time.time()
        # 转换为浮点类型，解决mean()函数的数据类型问题
        data_tensor = torch.from_numpy(data.astype(np.float32)).to(DEVICE)
        print(f"数据转移到GPU耗时: {time.time() - start_time:.2f}秒")
        
        # 使用PyTorch的unfold操作实现滑动窗口
        start_time = time.time()
        
        # 添加批次维度以使用unfold
        data_tensor = data_tensor.unsqueeze(1)  # [n_freqs, 1, height, width]
        
        # 对高度方向应用unfold
        unfolded_h = data_tensor.unfold(2, WINDOW_SIZE, STEP)  # [n_freqs, 1, h_out, width, window_size]
        
        # 对宽度方向应用unfold
        unfolded = unfolded_h.unfold(3, WINDOW_SIZE, STEP)  # [n_freqs, 1, h_out, w_out, window_size, window_size]
        
        # 计算窗口内的平均值
        result_tensor = unfolded.mean([-1, -2]).squeeze(1)  # [n_freqs, h_out, w_out]
        
        print(f"GPU滑动窗口计算耗时: {time.time() - start_time:.2f}秒")
        
        # 将结果转回CPU
        start_time = time.time()
        result = result_tensor.cpu().numpy()
        print(f"结果转回CPU耗时: {time.time() - start_time:.2f}秒")
        
        print(f"滑动窗口后数据形状: {result.shape}")
        
        # 验证形状是否符合预期
        expected_shape = (data.shape[0], PROCESSED_HEIGHT, PROCESSED_WIDTH)
        if result.shape != expected_shape:
            print(f"警告: 滑动窗口后形状 {result.shape} 与预期 {expected_shape} 不符")
        
        return result

class LorentzFitter:
    def __init__(self, background_init=20000.0):
        self.device = DEVICE
        # 修正频率点数计算
        num_freqs = FREQ_END - FREQ_START + 1
        self.freqs = torch.linspace(FREQ_START, FREQ_END, num_freqs, 
                                  device=self.device, dtype=torch.float32)
        print(f"频率点数: {num_freqs}")  # 调试信息
        
        # 使用传入的背景初始值
        self.init_params = torch.tensor([
            background_init,    # 背景
            2765, 600, 10,  # 峰1：中心频率，强度（正值，表示减去的量），半宽度
            2782, 600, 10,  # 峰2
            2816, 600, 10,  # 峰3
            2832, 600, 10,  # 峰4
            2916, 600, 10,  # 峰5
            2932, 600, 10,  # 峰6
            2950, 600, 10,  # 峰7
            2975, 600, 10,  # 峰8
        ], device=self.device)
        
        # 背景的边界也应该相应调整
        bg_lower = max(10000, background_init * 0.8)  # 背景下限设为初始值的70%或10000中的较大值
        bg_upper = background_init * 1.2  # 背景上限设为初始值的130%
        
        self.bounds_lower = np.array([
            bg_lower,       # 背景下限
            2750, 300, 1,  # 峰1下限
            2740, 300, 1,
            2780, 300, 1,
            2800, 300, 1,
            2860, 300, 1,
            2880, 300, 1,
            2920, 300, 1,
            2940, 300, 1
        ])
        
        self.bounds_upper = np.array([
            bg_upper,   # 背景上限
            2780, 800, 50,  # 峰1上限
            2800, 800, 50,
            2840, 800, 50, 
            2860, 800, 50,
            2920, 800, 50,
            2940, 800, 50,
            2980, 800, 50,
            3000, 800, 50
        ])

    def _lorentzian(self, params: torch.Tensor) -> torch.Tensor:
        """构建洛伦兹模型（8个峰向下）"""
        bg = params[0]
        # 确保freqs和数据维度匹配
        freqs = self.freqs  # [299]
        
        # 计算8个峰（注意这里是减法，表示向下的峰）
        peaks = []
        for i in range(8):  # 8个峰
            base_idx = 1 + i * 3
            center = params[base_idx]
            amp = params[base_idx + 1]  # 这里amp应该是正值
            width = params[base_idx + 2]
            # 修改为减法，表示从背景中减去峰
            peak = amp / (((freqs - center)/width)**2 + 1)
            peaks.append(peak)
            
        # 合并所有峰（从背景中减去）
        return bg - sum(peaks)

    # 添加背景估计函数
    def _estimate_background(self, spectrum, percentile=90):
        """估计频谱的背景噪声水平"""
        # 将数据转移到CPU进行处理
        if isinstance(spectrum, torch.Tensor):
            spectrum = spectrum.cpu().numpy()
            
        # 方法1: 使用高百分位数作为背景估计
        bg_estimate = np.percentile(spectrum, percentile)
        
        # 方法2: 寻找频谱中的平坦区域
        # 计算频谱的一阶差分
        diff = np.abs(np.diff(spectrum))
        # 找到差分较小的区域（平坦区域）
        flat_regions = np.where(diff < np.percentile(diff, 20))[0]
        if len(flat_regions) > 10:
            # 使用平坦区域的平均值作为背景
            flat_bg = np.mean(spectrum[flat_regions])
            # 取两种方法的平均值
            bg_estimate = (bg_estimate + flat_bg) / 2
        
        # 确保背景估计在合理范围内
        bg_estimate = max(bg_estimate, 10000)  # 设置最小背景值
        
        return bg_estimate

    def _fit_batch(self, batch: np.ndarray) -> np.ndarray:
        """使用Levenberg-Marquardt算法批量拟合，并对拟合结果进行排序"""
        print(f"输入batch形状: {batch.shape}")
        
        # 确保数据是正确的形状：[pixels, frequencies]
        if batch.shape[0] == len(self.freqs):
            batch = batch.transpose(1, 2, 0).reshape(-1, len(self.freqs))
        print(f"重塑后batch形状: {batch.shape}")
        
        # 将频率转换为NumPy数组用于scipy
        freqs_np = self.freqs.cpu().numpy()
        init_params_np = self.init_params.cpu().numpy()
        
        # 定义残差函数（用于LM算法）
        def residual_fn(params, y_data, x_data):
            # 背景
            bg = params[0]
            
            # 计算8个峰
            model = np.zeros_like(x_data)
            for i in range(8):
                base_idx = 1 + i * 3
                center = params[base_idx]
                amp = params[base_idx + 1]
                width = params[base_idx + 2]
                peak = amp / (((x_data - center)/width)**2 + 1)
                model += peak
            
            # 添加背景并减去峰（向下的峰）
            model = bg - model
            
            # 返回残差
            return model - y_data
        
        results = []
        
        # 使用多线程加速处理
        def process_pixel(pixel_data, pixel_idx):
            try:
                # 为每个像素估计背景值
                bg_estimate = self._estimate_background(pixel_data)
                
                # 创建该像素的初始参数，使用估计的背景值
                pixel_init_params = init_params_np.copy()
                pixel_init_params[0] = bg_estimate
                
                # 调整背景边界
                pixel_bounds_lower = self.bounds_lower.copy()
                pixel_bounds_upper = self.bounds_upper.copy()
                pixel_bounds_lower[0] = max(10000, bg_estimate * 0.8)
                pixel_bounds_upper[0] = bg_estimate * 1.2
                
                res = least_squares(
                    residual_fn, 
                    pixel_init_params,
                    bounds=(pixel_bounds_lower, pixel_bounds_upper),
                    args=(pixel_data, freqs_np),
                    method='trf',
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                    max_nfev=500,
                    loss='soft_l1',
                    verbose=0,
                    diff_step=1e-3
                )               
                    
                # 提取所有峰的中心频率
                centers = np.array([res.x[1 + j*3] for j in range(8)])
                
                # 对峰值中心频率进行排序
                sorted_centers = np.sort(centers)
                
                return sorted_centers, pixel_idx
            except Exception as e:
                print(f"拟合错误 (像素 {pixel_idx}): {str(e)}")
                return np.full(8, np.nan), pixel_idx
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(0, len(batch), GPU_BATCH):
                chunk = batch[i:i+GPU_BATCH]
                print(f"提交第 {i}/{len(batch)} 批次，形状: {chunk.shape}")
                
                for pixel_idx in range(len(chunk)):
                    futures.append(executor.submit(process_pixel, chunk[pixel_idx], i+pixel_idx))
            
            # 存储最后一次成功的拟合结果
            last_successful_centers = np.array([
                2765, 2782, 2816, 2832, 2916, 2932, 2950, 2975
            ])  # 初始值，以防第一次拟合就失败
            
            # 收集结果
            pixel_results = {}
            for future in tqdm(futures, desc="拟合进度"):
                result, pixel_idx = future.result()
                # 如果结果不包含NaN，更新最后一次成功的拟合结果
                if not np.any(np.isnan(result)):
                    last_successful_centers = result.copy()
                    pixel_results[pixel_idx] = result
                else:
                    # 对于失败的拟合，使用最后一次成功的结果
                    pixel_results[pixel_idx] = last_successful_centers.copy()
            
            # 按像素索引排序结果
            for i in range(len(batch)):
                if i in pixel_results:
                    results.append(pixel_results[i])
                else:
                    # 对于缺失的像素，也使用最后一次成功的结果
                    results.append(last_successful_centers.copy())
        
        results = np.array(results)
        print(f"拟合结果原始形状: {results.shape}")
        
        # 动态计算正确的形状，而不是硬编码
        reshaped_results = results.reshape(PROCESSED_HEIGHT, PROCESSED_WIDTH, 8)
        print(f"拟合结果重塑后形状: {reshaped_results.shape}")
        
        return reshaped_results

class MagneticFieldAnalyzer:
    def __init__(self):
        self.fitter = LorentzFitter()
    
    def axes_to_cartesian(self, axis_fields):
        """将轴向磁场转换为笛卡尔坐标系磁场 - GPU加速版本"""
        # 将数据转移到GPU
        start_time = time.time()
        axis_fields_tensor = torch.from_numpy(axis_fields).to(DEVICE)
        print(f"数据转移到GPU耗时: {time.time() - start_time:.2f}秒")
        
        # 分离每个NV轴向的两个峰值（正负方向）
        nv1_pos = axis_fields_tensor[..., 0]
        nv1_neg = axis_fields_tensor[..., 7]
        nv2_pos = axis_fields_tensor[..., 1]
        nv2_neg = axis_fields_tensor[..., 6]
        nv3_pos = axis_fields_tensor[..., 2]
        nv3_neg = axis_fields_tensor[..., 5]
        nv4_pos = axis_fields_tensor[..., 3]
        nv4_neg = axis_fields_tensor[..., 4]
        
        # 计算每个轴向的磁场大小（考虑正负方向）
        gyro_ratio = torch.tensor(2.87, device=DEVICE)  # 2.87 MHz/G 是旋磁比
        B1 = torch.abs(nv1_pos - nv1_neg) / gyro_ratio
        B2 = torch.abs(nv2_pos - nv2_neg) / gyro_ratio
        B3 = torch.abs(nv3_pos - nv3_neg) / gyro_ratio
        B4 = torch.abs(nv4_pos - nv4_neg) / gyro_ratio
        
        # 使用简化公式进行坐标转换
        sqrt6_4 = torch.tensor(np.sqrt(6)/4, device=DEVICE)
        sqrt3_4 = torch.tensor(np.sqrt(3)/4, device=DEVICE)
        
        Bx = sqrt6_4 * (B2 - B1)
        By = sqrt6_4 * (B3 - B4)
        Bz = sqrt3_4 * (B1 + B2 + B3 + B4)
        
        # 计算总磁场强度
        B_total = torch.sqrt(Bx**2 + By**2 + Bz**2)
        
        # 将结果转回CPU
        start_time = time.time()
        B_total_np = B_total.cpu().numpy()
        cartesian_fields_np = torch.stack([Bx, By, Bz], dim=-1).cpu().numpy()
        print(f"结果转回CPU耗时: {time.time() - start_time:.2f}秒")
        
        return B_total_np, cartesian_fields_np
    
    @staticmethod
    def enhance_snr(spectrum_data, method='wavelet'):
        """提高频谱数据的信噪比 - GPU加速版本"""
        # 获取数据维度
        n_freqs, height, width = spectrum_data.shape
        
        # 将数据转移到GPU
        start_time = time.time()
        data_tensor = torch.from_numpy(spectrum_data).to(DEVICE)
        print(f"数据转移到GPU耗时: {time.time() - start_time:.2f}秒")
        
        # 重塑数据为2D数组，每行是一个像素的频谱
        reshaped_data = data_tensor.permute(1, 2, 0).reshape(-1, n_freqs)  # [height*width, n_freqs]
        
        # 初始化结果张量
        enhanced_data = reshaped_data.clone()
        
        # 由于小波变换和Savitzky-Golay滤波在GPU上实现复杂，我们将这部分数据转回CPU处理
        if method == 'wavelet' or method == 'savgol' or method == 'combined':
            print("将数据转回CPU进行小波/SG滤波...")
            cpu_data = reshaped_data.cpu().numpy()
            
            if method == 'wavelet' or method == 'combined':
                print("应用小波去噪...")
                # 小波去噪（需要逐像素处理）
                wavelet = 'sym8'
                level = 3
                
                # 使用批处理加速
                batch_size = 1000
                for i in tqdm(range(0, cpu_data.shape[0], batch_size), desc="小波去噪"):
                    end_idx = min(i + batch_size, cpu_data.shape[0])
                    batch = cpu_data[i:end_idx]
                    
                    # 并行处理批次
                    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                        futures = []
                        for j in range(batch.shape[0]):
                            futures.append(executor.submit(
                                MagneticFieldAnalyzer._wavelet_denoise, 
                                batch[j], wavelet, level, n_freqs
                            ))
                        
                        # 收集结果
                        for j, future in enumerate(futures):
                            cpu_data[i+j] = future.result()
            
            if method == 'savgol' or method == 'combined':
                print("应用Savitzky-Golay滤波...")
                # Savitzky-Golay滤波
                window_length = 15  # 窗口长度必须是奇数
                polyorder = 3       # 多项式阶数
                
                # 向量化处理所有像素
                cpu_data = np.apply_along_axis(
                    lambda x: savgol_filter(x, window_length, polyorder), 
                    1, 
                    cpu_data
                )
            
            # 将处理后的数据转回GPU
            enhanced_data = torch.from_numpy(cpu_data).to(DEVICE)
        
        # 频域滤波可以在GPU上高效实现
        if method == 'frequency' or method == 'combined':
            print("应用频域滤波 (GPU)...")
            # 傅里叶变换
            signal_fft = torch.fft.rfft(enhanced_data, dim=1)
            
            # 设计频域滤波器（低通滤波器）
            cutoff = int(n_freqs * 0.2)  # 截止频率，保留20%的低频成分
            
            # 创建掩码
            mask = torch.ones(signal_fft.shape[1], device=DEVICE, dtype=torch.bool)
            mask[cutoff:] = False
            
            # 应用滤波器
            filtered_fft = signal_fft.clone()
            filtered_fft[:, ~mask] = 0
            
            # 逆傅里叶变换
            enhanced_data = torch.fft.irfft(filtered_fft, n=n_freqs, dim=1)
        
        # 重塑回原始形状
        enhanced_data = enhanced_data.reshape(height, width, n_freqs).permute(2, 0, 1)
        
        # 将结果转回CPU
        start_time = time.time()
        result = enhanced_data.cpu().numpy()
        print(f"结果转回CPU耗时: {time.time() - start_time:.2f}秒")
        
        return result
    
    @staticmethod
    def _wavelet_denoise(spectrum, wavelet, level, n_freqs):
        """对单个频谱进行小波去噪"""
        # 小波分解
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        # 阈值处理
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(spectrum)))
        
        # 应用软阈值
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        # 小波重构
        denoised_spectrum = pywt.waverec(coeffs_thresholded, wavelet)
        
        # 确保长度一致
        denoised_spectrum = denoised_spectrum[:n_freqs]
        
        return denoised_spectrum

    def analyze(self, ref_data: np.ndarray, sample_data: np.ndarray) -> np.ndarray:
        """执行完整分析流程 - GPU加速版本"""
        print(f"参考数据原始维度: {ref_data.shape}")
        print(f"样本数据原始维度: {sample_data.shape}")
        
        # 验证输入形状
        expected_shape = (FREQ_END - FREQ_START + 1, IMAGE_HEIGHT, IMAGE_WIDTH)
        if ref_data.shape != expected_shape or sample_data.shape != expected_shape:
            print(f"警告: 输入数据形状与预期 {expected_shape} 不符")
        
        # 空间降采样
        print("正在进行空间降采样...")
        ref_processed = SpatialProcessor.sliding_window(ref_data)
        sample_processed = SpatialProcessor.sliding_window(sample_data)
        
        print(f"参考数据降采样后维度: {ref_processed.shape}")
        print(f"样本数据降采样后维度: {sample_processed.shape}")
        
        # 添加信噪比提高处理
        print("正在提高信噪比...")
        ref_enhanced = self.enhance_snr(ref_processed, method='combined')
        sample_enhanced = self.enhance_snr(sample_processed, method='combined')
        print("信噪比提高完成")
        
        # 添加频谱拟合可视化
        analyzer = MagneticFieldAnalyzer()
        
        # 在分析前选择一些点进行拟合测试
        test_points = [
            (PROCESSED_HEIGHT//4, PROCESSED_WIDTH//4),
            (PROCESSED_HEIGHT//2, PROCESSED_WIDTH//2),
            (3*PROCESSED_HEIGHT//4, 3*PROCESSED_WIDTH//4)
        ]
        
        # 可视化每个测试点的频谱和拟合
        for i, (y, x) in enumerate(test_points):
            # 获取点的频谱（原始和增强后的）
            ref_spectrum_orig = ref_processed[:, y, x]
            sample_spectrum_orig = sample_processed[:, y, x]
            ref_spectrum = ref_enhanced[:, y, x]
            sample_spectrum = sample_enhanced[:, y, x]
            
            # 绘制频谱对比图
            plt.figure(figsize=(15, 8))
            
            # 原始频谱
            plt.subplot(2, 1, 1)
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(ref_spectrum_orig)), ref_spectrum_orig, 'b-', label='参考 (原始)')
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(sample_spectrum_orig)), sample_spectrum_orig, 'r-', label='样本 (原始)')
            plt.xlabel('频率 (MHz)')
            plt.ylabel('强度')
            plt.title(f'测试点 {i+1} ({x}, {y}) 的原始频谱')
            plt.legend()
            
            # 增强后的频谱
            plt.subplot(2, 1, 2)
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(ref_spectrum)), ref_spectrum, 'b-', label='参考 (增强)')
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(sample_spectrum)), sample_spectrum, 'r-', label='样本 (增强)')
            plt.xlabel('频率 (MHz)')
            plt.ylabel('强度')
            plt.title(f'测试点 {i+1} ({x}, {y}) 的增强频谱')
            plt.legend()
            
            plt.tight_layout()
            save_dir = os.path.expanduser("~/Desktop")
            plt.savefig(os.path.join(save_dir, f"test_point_{i+1}_spectrum_comparison.png"), dpi=300)
            plt.close()
            
            # 单独保存增强后的频谱图（与原代码保持一致）
            plt.figure(figsize=(12, 6))
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(ref_spectrum)), ref_spectrum, 'b-', label='参考 (增强)')
            plt.plot(np.linspace(FREQ_START, FREQ_END, len(sample_spectrum)), sample_spectrum, 'r-', label='样本 (增强)')
            plt.xlabel('频率 (MHz)')
            plt.ylabel('强度')
            plt.title(f'测试点 {i+1} ({x}, {y}) 的增强频谱')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"test_point_{i+1}_spectrum.png"), dpi=300)
            plt.close()
        
        # 分别为参考数据和样本数据创建拟合器，使用不同的背景初始值
        # 使用增强后的数据进行拟合
        ref_fitter = LorentzFitter(background_init=35000.0)
        sample_fitter = LorentzFitter(background_init=32500.0)
        
        # 分别进行拟合
        print("开始拟合参考数据...")
        ref_params = ref_fitter._fit_batch(ref_enhanced)
        print("开始拟合样本数据...")
        sample_params = sample_fitter._fit_batch(sample_enhanced)
        
        # 修改：分别计算参考样本和测试样本的磁场，然后计算差值
        print("计算参考样本磁场...")
        ref_total_field, ref_cartesian_fields = self.axes_to_cartesian(ref_params)
        
        print("计算测试样本磁场...")
        sample_total_field, sample_cartesian_fields = self.axes_to_cartesian(sample_params)
        
        print("计算磁场差值...")
        # 计算磁场差值
        delta_total_field = sample_total_field - ref_total_field
        delta_cartesian_fields = sample_cartesian_fields - ref_cartesian_fields
        
        # 保存参考和样本的原始磁场数据，便于后续分析
        save_dir = os.path.expanduser("~/Desktop")
        np.save(os.path.join(save_dir, "ref_total_field.npy"), ref_total_field)
        np.save(os.path.join(save_dir, "ref_cartesian_fields.npy"), ref_cartesian_fields)
        np.save(os.path.join(save_dir, "sample_total_field.npy"), sample_total_field)
        np.save(os.path.join(save_dir, "sample_cartesian_fields.npy"), sample_cartesian_fields)
        
        # 可视化参考和样本的磁场
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(ref_total_field, cmap='jet')
        plt.colorbar(label="磁场强度 (G)")
        plt.title("参考样本磁场")
        
        plt.subplot(1, 3, 2)
        plt.imshow(sample_total_field, cmap='jet')
        plt.colorbar(label="磁场强度 (G)")
        plt.title("测试样本磁场")
        
        plt.subplot(1, 3, 3)
        plt.imshow(delta_total_field, cmap='jet')
        plt.colorbar(label="磁场差值 (G)")
        plt.title("磁场差值")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "magnetic_field_comparison.png"), dpi=300)
        plt.close()
        
        return delta_total_field, delta_cartesian_fields

def main():
    try:
        # 记录开始时间
        start_time = time.time()
        
        # 确保目录存在 - 在函数开始时定义save_dir
        save_dir = os.path.expanduser("~/Desktop")
        
        # 加载参考样本和测试样本
        ref_folder = "20250219偏置磁场"
        sample_folder = "20250219铷铁硼偏置磁场"
        
        print(f"正在加载参考数据: {ref_folder}")
        ref_data = DataLoader.load_folder(ref_folder)
        print(f"正在加载测试数据: {sample_folder}")
        sample_data = DataLoader.load_folder(sample_folder)
        
        # 检查数据有效性
        print(f"参考数据统计: 最小值={np.min(ref_data)}, 最大值={np.max(ref_data)}, 平均值={np.mean(ref_data)}")
        print(f"样本数据统计: 最小值={np.min(sample_data)}, 最大值={np.max(sample_data)}, 平均值={np.mean(sample_data)}")
        
        # 可视化原始数据中的一个频率切片
        middle_freq_idx = (FREQ_END - FREQ_START) // 2
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(ref_data[middle_freq_idx], cmap='viridis')
        plt.colorbar(label="强度")
        plt.title("参考数据中频切片")
        
        plt.subplot(1, 2, 2)
        plt.imshow(sample_data[middle_freq_idx], cmap='viridis')
        plt.colorbar(label="强度")
        plt.title("样本数据中频切片")
        
        plt.savefig(os.path.join(save_dir, "raw_data_check.png"), dpi=300)
        
        # 执行分析
        analyzer = MagneticFieldAnalyzer()
        delta_total_field, delta_cartesian_fields = analyzer.analyze(ref_data, sample_data)
        
        # 可视化结果
        print("正在生成磁场强度分布图...")
        plt.figure(figsize=(12, 8))
        plt.imshow(delta_total_field, cmap='jet', 
                  extent=[0, PROCESSED_WIDTH, 0, PROCESSED_HEIGHT])
        plt.colorbar(label="总磁场强度 (G)")
        plt.title("磁场强度分布")
        
        plt.savefig(os.path.join(save_dir, "total_magnetic_field.png"), 
                   dpi=300, bbox_inches='tight')
        
        # 可视化各个分量
        field_components = ["Bx", "By", "Bz"]
        for i, component in enumerate(field_components):
            print(f"正在生成{component}分量分布图...")
            plt.figure(figsize=(12, 8))
            plt.imshow(delta_cartesian_fields[..., i], cmap='jet', 
                      extent=[0, PROCESSED_WIDTH, 0, PROCESSED_HEIGHT])
            plt.colorbar(label=f"{component} (G)")
            plt.title(f"磁场{component}分量分布")
            plt.savefig(os.path.join(save_dir, f"magnetic_field_{component}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 保存处理后的数据
        print("正在保存处理后的数据...")
        np.save(os.path.join(save_dir, "total_field.npy"), delta_total_field)
        np.save(os.path.join(save_dir, "cartesian_fields.npy"), delta_cartesian_fields)
        
        # 生成磁场强度直方图
        print("正在生成磁场强度直方图...")
        plt.figure(figsize=(10, 6))
        plt.hist(delta_total_field.flatten(), bins=50, alpha=0.7)
        plt.xlabel("磁场强度 (G)")
        plt.ylabel("像素数量")
        plt.title("磁场强度分布直方图")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "magnetic_field_histogram.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成磁场方向分布图（使用箭头表示）
        print("正在生成磁场方向分布图...")
        plt.figure(figsize=(14, 10))
        
        # 降采样以避免箭头过密
        step = 20
        Y, X = np.mgrid[0:PROCESSED_HEIGHT:step, 0:PROCESSED_WIDTH:step]
        
        # 提取磁场分量并归一化
        Bx_sampled = delta_cartesian_fields[::step, ::step, 0]
        By_sampled = delta_cartesian_fields[::step, ::step, 1]
        Bz_sampled = delta_cartesian_fields[::step, ::step, 2]
        
        # 计算平面内的磁场强度用于归一化
        B_planar = np.sqrt(Bx_sampled**2 + By_sampled**2)
        
        # 绘制背景为总磁场强度
        plt.imshow(delta_total_field, cmap='jet', 
                  extent=[0, PROCESSED_WIDTH, 0, PROCESSED_HEIGHT],
                  alpha=0.7)
        
        # 绘制箭头表示磁场方向
        plt.quiver(X, Y, Bx_sampled, By_sampled, 
                  scale=50, color='white', width=0.003, 
                  headwidth=3, headlength=4, headaxislength=3)
        
        plt.colorbar(label="总磁场强度 (G)")
        plt.title("磁场方向分布")
        plt.savefig(os.path.join(save_dir, "magnetic_field_direction.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成3D磁场可视化
        print("正在生成3D磁场可视化...")
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 降采样以提高性能
        step = 30
        Y, X = np.mgrid[0:PROCESSED_HEIGHT:step, 0:PROCESSED_WIDTH:step]
        Z = np.zeros_like(X)
        
        # 提取磁场分量
        Bx_sampled = delta_cartesian_fields[::step, ::step, 0]
        By_sampled = delta_cartesian_fields[::step, ::step, 1]
        Bz_sampled = delta_cartesian_fields[::step, ::step, 2]
        
        # 绘制3D箭头
        ax.quiver(X, Y, Z, Bx_sampled, By_sampled, Bz_sampled, 
                 length=5, normalize=True, color='b', alpha=0.7)
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D磁场分布')
        
        # 调整视角
        ax.view_init(elev=30, azim=45)
        
        plt.savefig(os.path.join(save_dir, "magnetic_field_3d.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算并显示总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"分析完成！结果已保存至桌面。")
        print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        
        # 保存性能数据
        with open(os.path.join(save_dir, "performance_log.txt"), "w") as f:
            f.write(f"使用设备: {DEVICE}\n")
            if torch.cuda.is_available():
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")
            f.write(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒\n")
            f.write(f"处理的数据尺寸: {ref_data.shape}\n")
            f.write(f"处理后的数据尺寸: {delta_total_field.shape}\n")
            f.write(f"线程数: {NUM_WORKERS}\n")
            f.write(f"GPU批处理大小: {GPU_BATCH}\n")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 即使出错也保存已完成的部分
        end_time = time.time()
        total_time = end_time - start_time
        print(f"程序运行了 {total_time:.2f} 秒后出错")

if __name__ == "__main__":
    main()