import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional

from config.settings import settings
from config.logger_config import get_logger, log_info, log_warning, log_error

# 获取当前logger
logger = get_logger(__name__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_PATH)

# 模型地址
BGE_SMALL_ZH_PATH = os.path.join(ROOT_PATH, f"model/{settings.supported_models.get('bge_small_zh_v1.5')}")
BGE_SMALL_EN_PATH = os.path.join(ROOT_PATH, f"model/{settings.supported_models.get('bge_small_en_v1.5')}")

# 加载模型
try:
    log_info("开始加载嵌入模型：bge-small-zh-v1.5")

    BGE_SMALL_ZH_MODEL = SentenceTransformer(
        model_name_or_path=BGE_SMALL_ZH_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    log_info("模型加载成功！")
except Exception as e:
    log_error(f"模型加载失败：{str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

try:
    log_info("开始加载嵌入模型：bge-small-en-v1.5")

    BGE_SMALL_EN_MODEL = SentenceTransformer(
        model_name_or_path=BGE_SMALL_EN_PATH,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    log_info("模型加载成功！")
except Exception as e:
    log_error(f"模型加载失败：{str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")


def bge_embedding(texts: Union[str, List[str]], model_name):
    if isinstance(texts, str):
        contexts = [texts[:settings.max_squentence_length]]
    else:
        contexts = [text[:settings.max_squentence_length] for text in texts]

    embedding_res = []

    if model_name == "bge-small-zh-v1.5":
        try:
            log_info(f"开始使用{model_name}进行文本向量化......")
            embedding_res = BGE_SMALL_ZH_MODEL.encode(contexts)
            log_info("文本向量化完成")
        except Exception as e:
            log_error(f"文本向量化执行出错：{str(e)}")
            raise RuntimeError(f"Failed to embedding: {str(e)}")
    elif model_name == "bge-small-en-v1.5":
        try:
            log_info(f"开始使用{model_name}进行文本向量化......")
            embedding_res = BGE_SMALL_EN_MODEL.encode(contexts)
            log_info("文本向量化完成")
        except Exception as e:
            log_error(f"文本向量化执行出错：{str(e)}")
            raise RuntimeError(f"Failed to embedding: {str(e)}")

    return embedding_res

if __name__ == "__main__":
    text = "hello"
    text2 = '你好'
    print(bge_embedding(text2, "bge-small-zh-v1.5"))
    print(bge_embedding(text, "bge-small-en-v1.5"))


