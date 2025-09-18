from flask import Flask, request, jsonify

from model_factory.embedder import bge_loader
from config.logger_config import get_logger, log_error, log_info


# 获取挡墙logger
logger = get_logger("server")

app = Flask(__name__)


@app.route('/embedding', methods=['POST', "GET"])
def embedding():
    req_data = request.get_json()
    contexts = req_data.get("contexts", None)
    model_name = req_data.get("model_name", None)
    if not contexts:
        log_error("No contexts provided")
    try:
        log_info(f"开始进行文本向量化....")
        embedding_res = bge_loader.encode_text(contexts, model_name)
        log_info(f"文本向量化已完成......")
        response = {
            "status": 200,
            "embedding_res": embedding_res
        }
        return jsonify(response)
    except Exception as e:
        log_error(f"文本向量化失败：{str(e)}")
        response = {
            "status": 400,
            "embedding_res": []
        }
        return jsonify(response)


@app.route('/get_available_models')
def get_available_models():
    """获取可用模型"""
    return jsonify(bge_loader.get_available_models())


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        "status": 200,
        "message": "service is healthy!"
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)