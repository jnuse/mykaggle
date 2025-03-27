import cv2
import numpy as np
import os
import glob

def msrcr_enhancement(
    img,
    scales=[15, 80, 250],  # 多尺度高斯核大小
    alpha=125,              # 增益参数
    beta=46,                # 偏移参数
    color_restore=1.0       # 色彩恢复系数
):
    # 转换到Lab颜色空间并转换为浮点型
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # 多尺度Retinex处理（增强亮度通道）
    l_retinex = np.zeros_like(l_channel)
    for scale in scales:
        # 确保高斯核尺寸为正奇数
        if scale % 2 == 0 or scale <= 0:
            scale = max(3, scale + 1 if scale % 2 == 0 else scale)
        l_blur = cv2.GaussianBlur(l_channel, (scale, scale), 0)
        l_retinex += np.log(l_channel + 1) - np.log(l_blur + 1)
    l_retinex = l_retinex / len(scales)  # 取平均
    
    # 色彩恢复（调整a、b通道）
    a_restored = a_channel * color_restore
    b_restored = b_channel * color_restore
    
    # 关键修复：统一数据类型和范围
    # 1. 处理亮度通道（应用增益/偏移并转为uint8）
    l_retinex = alpha * l_retinex + beta
    l_retinex = np.clip(l_retinex, 0, 255).astype(np.uint8)
    
    # 2. 处理色彩通道（裁剪到有效范围并转为uint8）
    a_restored = np.clip(a_restored, 0, 255).astype(np.uint8)
    b_restored = np.clip(b_restored, 0, 255).astype(np.uint8)
    
    # 验证通道一致性
    assert l_retinex.shape == a_restored.shape == b_restored.shape, "通道尺寸不一致"
    assert l_retinex.dtype == a_restored.dtype == b_restored.dtype == np.uint8, "数据类型不一致"
    
    # 合并通道并转换回BGR
    enhanced_lab = cv2.merge([l_retinex, a_restored, b_restored])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def batch_msrcr(input_folder, output_folder, **kwargs):
    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(input_folder, "*.[jpJP][pnPN]*[gG]"))
    
    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: 无法读取 {image_path}")
            continue
        
        # 应用MSRCR增强
        enhanced_img = msrcr_enhancement(img, **kwargs)
        
        # 保存结果
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, enhanced_img)
        print(f"Processed: {filename}")

if __name__ == "__main__":
    input_folder = "../testA"
    output_folder = "../output/testA_msrcr"
    
    # 参数配置（可根据需求调整）
    params = {
        "scales": [15, 80, 250],
        "alpha": 125,
        "beta": 46,
        "color_restore": 1.0
    }
    
    batch_msrcr(input_folder, output_folder, **params)