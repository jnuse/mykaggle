import os
import glob
import time
import shutil
from pathlib import Path
from WaterNet.WaterNet_test import WaterNet_test
from UWCNN.UWCNN_test import UWCNN_test

def ensure_dir(directory):
    """创建目录（如果不存在）"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def batch_process(input_folder, output_folder, modes=['UWCNN', 'WaterNet']):
    # 创建输出根目录
    ensure_dir(output_folder)
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(input_folder, "*.[jpJP][pnPN]*[gG]"))
    total_images = len(image_files)
    
    if total_images == 0:
        print("No images found in input folder.")
        return
    
    # 初始化计时
    total_time = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\nProcessing: {filename}")
        
        # 为每个模型处理图像
        for mode in modes:
            # 创建模型专属输入缓存和输出目录
            model_input_dir = os.path.join("input_cache", mode)
            model_output_dir = os.path.join(output_folder, mode)
            ensure_dir(model_input_dir)
            ensure_dir(model_output_dir)
            
            # 准备输入缓存文件
            input_cache_path = os.path.join(model_input_dir, filename)
            shutil.copy(image_path, input_cache_path)
            
            # 调用测试函数（需预先定义UWCNN_test和WaterNet_test）
            try:
                start_time = time.time()
                if mode == 'UWCNN':
                    output_name = UWCNN_test(filename)  # 假设返回处理后的文件名
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    # 移动结果到输出目录
                    src_path = os.path.join(mode, "/kaggle/working/UIE-DL/UWCNN/output", output_name)  # 假设模型输出到此路径
                    dst_path = os.path.join(model_output_dir, output_name)
                    shutil.move(src_path, dst_path)
                    print(f"  {mode} done. Time: {elapsed_time:.2f}s")
                    
                    # 清理输入缓存
                    os.remove(input_cache_path)
                elif mode == 'WaterNet':
                    output_name = WaterNet_test(filename)
                    elapsed_time = time.time() - start_time
                    total_time += elapsed_time
                    # 移动结果到输出目录
                    src_path = os.path.join(mode, "/kaggle/working/UIE-DL/WaterNet/output", output_name)  # 假设模型输出到此路径
                    dst_path = os.path.join(model_output_dir, output_name)
                    shutil.move(src_path, dst_path)
                    print(f"  {mode} done. Time: {elapsed_time:.2f}s")
                    
                    # 清理输入缓存
                    os.remove(input_cache_path)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                

                
            except Exception as e:
                print(f"  Error in {mode}: {str(e)}")
                continue
    
    print(f"\nAll {total_images} images processed.")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results saved to: {output_folder}")

if __name__ == "__main__":
    input_folder = "../testA"
    output_folder = "../output/testA_uie"
    
    # 同时处理两个模型
    batch_process(input_folder, output_folder, modes=['UWCNN', 'WaterNet'])










# import streamlit as st 
# import os
# from WaterNet.WaterNet_test import WaterNet_test
# from UWCNN.UWCNN_test import UWCNN_test
# import time

# # streamlit run app.py

# # 设置缓存文件夹
# cache_dir = 'cache'
# if not os.path.isdir(cache_dir):
#     os.makedirs(cache_dir)

# # 设置全局属性
# st.set_page_config(
#     page_title='水下图像增强系统',
#     page_icon=' ',
#     layout='wide'
# )

# # 设置侧边栏
# with st.sidebar:
#     st.title('欢迎来到水下图像增强系统')
#     file_uploader = st.file_uploader(label='上传所需处理的图片', type=['.png','.bmp','jpg'])
#     st.markdown('---')
#     mode = st.radio(label='模型', options=['UWCNN', 'WaterNet'])
#     detect = st.button(label='开始处理')

# # 设置文件加载
# if file_uploader:
#     file_name = file_uploader.name
#     input_cache_path = os.path.join(cache_dir, file_name)
#     open(input_cache_path, 'wb').write(file_uploader.read())

# # 设置展示框
# p1, p2 = st.columns(spec=2)
# p1.title('Input-picture')
# p2.title('Output-picture')

# # 显示加载的图片
# if file_uploader and os.path.exists(input_cache_path):
#     p1.image(image=input_cache_path, caption='input image', use_column_width='always')
# else:
#     p1.info('等待图片加载...')

# # 开始检测
# if detect and os.path.exists(input_cache_path):
#     #调用test程序
#     start_time = time.time()
#     if mode == 'UWCNN':
#         name = UWCNN_test(file_name)
#         p2.image(image=f'UWCNN\\output\\{name}', caption='output image', use_column_width='always')
#     elif mode == 'WaterNet':
#         name = WaterNet_test(file_name)
#         p2.image(image=f'WaterNet\\output\\{name}', caption='output image', use_column_width='always')
#     end_time = time.time()
#     last_time = '%.2f'%(end_time-start_time)
#     st.info(f'处理耗时:{last_time}s')
# else:
#     p2.info('等待处理...')

