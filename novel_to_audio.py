import json
import os
import torch
import torchaudio
from GPT_SoVITS.inference_webui import (
    get_tts_wav,
    change_gpt_weights,
    change_sovits_weights,
    init_bigvgan,
    init_hifigan,
    DictToAttrRecursive
)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return DictToAttrRecursive(config)

def process_novel(novel_path, config):
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 初始化模型
    change_gpt_weights(config.gpt_path)
    change_sovits_weights(config.sovits_path)
    init_bigvgan()
    init_hifigan()
    
    # 读取小说内容
    with open(novel_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按段落分割文本
    # paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    # paragraphs = [content.strip()]
    
    # 处理每个段落
    for i, paragraph in enumerate(paragraphs):
        print(f"正在处理第 {i+1}/{len(paragraphs)} 段...")
        
        # 生成音频
        audio_generator = get_tts_wav(
            ref_wav_path=config.ref_wav_path,
            prompt_text=config.prompt_text,
            prompt_language=config.prompt_language,
            text=paragraph,
            text_language=config.text_language,
            how_to_cut=config.how_to_cut,
            top_k=config.top_k,
            top_p=config.top_p,
            temperature=config.temperature,
            ref_free=config.ref_free,
            speed=config.speed,
            if_freeze=config.if_freeze,
            sample_steps=config.sample_steps,
            if_sr=config.if_sr,
            pause_second=config.pause_second
        )
        
        # 从生成器中获取采样率和音频数据
        sample_rate, audio_data = next(audio_generator)
        
        # 将numpy数组转换为torch张量
        audio_tensor = torch.from_numpy(audio_data).float() / 32767.0
        
        # 保存音频文件
        output_path = os.path.join(config.output_dir, f"chapter_{i+1:04d}.wav")
        torchaudio.save(output_path, audio_tensor.unsqueeze(0), sample_rate)
        print(f"已保存到: {output_path}")

def main():
    # 加载配置
    config = load_config('infer.json')
    
    # 检查必要的文件是否存在
    if not os.path.exists(config.ref_wav_path):
        raise FileNotFoundError(f"参考音频文件不存在: {config.ref_wav_path}")
    
    if not os.path.exists(config.novel_path):
        raise FileNotFoundError(f"小说文件不存在: {config.novel_path}")
    
    # 处理小说
    process_novel(config.novel_path, config)
    print("处理完成！")

if __name__ == "__main__":
    main() 