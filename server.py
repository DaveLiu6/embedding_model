from flask import Flask, request, jsonify

from model_factory.bge_embedder import bge_embedding
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
        embedding_res = bge_embedding(contexts, model_name)
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)