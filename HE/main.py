import cv2
import os
import glob

def apply_histogram_equalization(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中所有图片文件
    image_files = glob.glob(os.path.join(input_folder, "*.*"))
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    
    for image_path in image_files:
        # 检查文件扩展名是否为支持的格式
        if os.path.splitext(image_path)[1].lower() not in supported_formats:
            continue
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: 无法读取 {image_path}")
            continue
        
        # 判断是否为彩色图像
        if len(img.shape) == 3 and img.shape[2] == 3:
            # 转换为YUV颜色空间，仅对亮度通道进行均衡化
            # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # equ = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            equ = cv2.equalizeHist(img)
        else:
            # 灰度图像直接均衡化
            equ = cv2.equalizeHist(img)
        
        # 构造输出路径
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)
        
        # 保存处理后的图像
        cv2.imwrite(output_path, equ)
        print(f"Processed: {filename}")

if __name__ == "__main__":
    input_folder = "../testA"      # 输入文件夹路径
    output_folder = "../output/testA_he"  # 输出文件夹路径
    
    apply_histogram_equalization(input_folder, output_folder)