import os
from typing import Dict, List


class Settings:
    """应用配置"""
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 模型配置
    default_model: str = "bge-small-zh-v1.5"
    max_squentence_length: int = 512
    normalize_embeddings: bool = True

    # 支持模型列表
    supported_models: Dict[str, str] = {
        "bge_small_zh_v1.5": "bge-small-zh-v1.5",
        "bge_small_en_v1.5": "bge-small-en-v1.5",
        "qwen3_embedding_0.6b": "qwen3-embedding-0.6b"
    }

    # 性能配置
    batch_size: int = 32
    device: str = 'cpu'  # auto, cpu, cuda


settings = Settings()

print(settings.ROOT_PATH)
