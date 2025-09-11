import logging
import os
from logging import handlers
from pathlib import Path


class LoggerSetup:
    """
    单例模式：整个应用使用一套日志配置
    文件轮转：当日志文件超过10M时自动穿件新文件
    """
    _isInitialized = False

    def __init__(self):
        if not self._isInitialized:
            self.setup_logger()
            self._isInitialized = True

    def setup_logger(self):
        # 创建log目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # 配置参数
        log_file = os.path.join(log_dir, 'test.log')
        max_bytes = 10 * 1024 * 1024   # 10M per file
        backup_count = 10  # 保留10个备份文件
        long_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # 创建根logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # 清除所有现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 创建格式处理器
        formatter = logging.Formatter(long_format, date_format)

        # 文件处理器（带轮转功能）
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


def get_logger(name=None):
    """获取logger实例"""
    LoggerSetup()
    return logging.getLogger(name or __name__)

# 便捷日志函数
def log_debug(message, logger_name=None):
    get_logger(logger_name).debug(message)

def log_info(message, logger_name=None):
    get_logger(logger_name).info(message)

def log_warning(message, logger_name=None):
    get_logger(logger_name).warning(message)

def log_error(message, logger_name=None):
    get_logger(logger_name).error(message)

def log_critical(message, logger_name=None):
    get_logger(logger_name).critical(message)
