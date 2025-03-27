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
    # 转换到Lab颜色空间
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # 多尺度Retinex处理
    l_retinex = np.zeros_like(l_channel)
    for scale in scales:
        l_blur = cv2.GaussianBlur(l_channel, (scale, scale), 0)
        l_retinex += np.log(l_channel + 1) - np.log(l_blur + 1)
    l_retinex /= len(scales)
    
    # 色彩恢复（调整a、b通道）
    a_restored = a_channel * color_restore
    b_restored = b_channel * color_restore
    
    # 合并通道并应用增益/偏移
    l_retinex = alpha * l_retinex + beta
    l_retinex = np.clip(l_retinex, 0, 255).astype(np.uint8)
    
    enhanced_lab = cv2.merge([l_retinex, a_restored, b_restored])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_Lab2BGR)
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