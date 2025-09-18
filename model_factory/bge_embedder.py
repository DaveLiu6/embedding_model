import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional

from config.settings import settings
from config.logger_config import get_logger, log_info, log_warning, log_error

# 获取当前logger
logger = get_logger(__name__)

# 模型地址
BGE_SMALL_ZH_PATH = os.path.join(settings.ROOT_PATH, f"model/{settings.supported_models.get('bge_small_zh_v1.5')}")
BGE_SMALL_EN_PATH = os.path.join(settings.ROOT_PATH, f"model/{settings.supported_models.get('bge_small_en_v1.5')}")


class BGEEmbedding:
    def __init__(self):
        """"初始化，并加载所有模型"""
        # 默认支持的模型
        if not settings.supported_models:
            log_warning("No support models")
        else:
            self.supported_models = settings.supported_models

        # 存储加载的模型
        self.models: Dict[str, SentenceTransformer] = {}
        self.model_status: Dict[str, bool] = {}

        # 依次加载所有支持模型
        self._load_all_models()

    def _load_all_models(self):
        """加载所有支持模型"""
        log_info("开始加载嵌入模型......")

        for alias, model_name in self.supported_models.items():
            try:
                log_info(f"正在加载模型：{model_name}")
                model_path = os.path.join(settings.ROOT_PATH, f"model/{settings.supported_models.get(alias)}")
                model = SentenceTransformer(
                    model_name_or_path=model_path,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                self.models[model_name] = model
                self.model_status[model_name] = True
                log_info(f"模型{model_name}加载成功！")
            except Exception as e:
                self.model_status[model_name] = False
                log_error(f"模型{model_name}加载失败：{e}")

        # 输出模型加载详情
        loaded_count = sum(self.model_status.values())
        total_count = len(self.supported_models)
        log_info(f"模型加载完成：{loaded_count}个加载成功，{total_count - loaded_count}个加载失败")

    def get_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """
        获取指定模型
        :param model_name: 模型名称
        :return: SentenceTransformer模型对象，如果模型未加载，则返回None
        """
        if model_name not in self.models:
            log_error(f"模型{model_name}未找到或加载失败")
            return None
        return self.models[model_name]

    def encode_text(self, texts: Union[str, List[str]], model_name: str) -> Optional[list]:
        """
        使用指定模型对文本进行编码
        :param text: 要编码的文本，支持str 和 list
        :param model_name: 模型名称
        :return: 编码后的向量，若模型不可用返回空
        """
        model = self.get_model(model_name)
        if not model:
            return None

        if isinstance(texts, str):
            texts = [texts[: settings.max_squentence_length]]
        else:
            texts = [text[: settings.max_squentence_length] for text in texts]

        try:
            log_info(f"开始使用模型{model_name}对文本编码")
            embeddings_res = model.encode(texts).tolist()
            log_info(f"文本编码成功！")
            return embeddings_res
        except Exception as e:
            log_info(f"文本编码失败：{e}")
            return None

    def get_available_models(self) -> Dict[str, bool]:
        """获取所有模型的可用状态"""
        return self.model_status.copy()

    def is_model_available(self, model_name: str) -> bool:
        return self.model_status.get(model_name, False)

bge_loader = BGEEmbedding()

if __name__ == "__main__":

    print(bge_loader.get_available_models())
    print(bge_loader.is_model_available("bge-small-zh-v1.5"))
    text1 = 'hello'
    text2 = ["今天天气真好", "你好"]
    print(bge_loader.encode_text(text1, model_name="bge-small-en-v1.5"))
    print(bge_loader.encode_text(text2, model_name="bge-small-zh-v1.5"))


