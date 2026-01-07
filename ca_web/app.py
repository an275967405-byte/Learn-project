"""
Flask应用主入口
"""
from flask import Flask
from config import DEBUG, HOST, PORT, CORS_ORIGIN, CORS_HEADERS, CORS_METHODS
from routes import student_bp, static_bp


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    
    # 注册蓝图
    app.register_blueprint(student_bp)
    app.register_blueprint(static_bp)
    
    # 配置CORS
    @app.after_request
    def add_cors_headers(response):
        response.headers.add("Access-Control-Allow-Origin", CORS_ORIGIN)
        response.headers.add('Access-Control-Allow-Headers', CORS_HEADERS)
        response.headers.add('Access-Control-Allow-Methods', CORS_METHODS)
        return response
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=DEBUG, host=HOST, port=PORT)

