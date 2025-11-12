"""
Nano-vLLM 基本使用示例

这个示例展示了如何使用 Nano-vLLM 进行批量文本生成推理：
1. 加载模型和分词器
2. 配置采样参数
3. 准备提示词
4. 执行批量推理
5. 输出结果
"""

import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # 1. 模型路径配置
    # 指定本地模型路径，使用 expanduser 支持 ~ 符号
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    
    # 2. 加载分词器
    # 使用 Hugging Face 的分词器来处理文本
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    # 3. 初始化 LLM 引擎
    # - enforce_eager=True: 禁用 CUDA graph 优化，使用即时执行模式（便于调试）
    # - tensor_parallel_size=1: 不使用张量并行，单 GPU 推理
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 4. 配置采样参数
    # - temperature=0.6: 控制生成的随机性（越高越随机，越低越确定）
    # - max_tokens=256: 每个请求最多生成 256 个 token
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 5. 准备提示词列表
    # 定义两个测试问题
    prompts = [
        "introduce yourself",  # 自我介绍
        "list all prime numbers within 100",  # 列出 100 以内的质数
    ]
    
    # 6. 应用聊天模板
    # 将普通文本转换为符合模型训练时的对话格式
    # - role="user": 表示这是用户的输入
    # - tokenize=False: 返回文本而非 token IDs
    # - add_generation_prompt=True: 添加模型生成回复的提示符
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    
    # 7. 批量推理
    # 将所有 prompts 一次性发送给模型进行批处理推理
    # 这比逐个处理更高效，可以充分利用 GPU 并行能力
    outputs = llm.generate(prompts, sampling_params)

    # 8. 输出结果
    # 遍历每个提示词和对应的生成结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")  # 显示原始提示词
        print(f"Completion: {output['text']!r}")  # 显示生成的文本


if __name__ == "__main__":
    main()
