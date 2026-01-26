from flask_ import NBA_bp
from flask import Flask
from scheduler_dir.scheduler_extract import nba_scheduler
import os

def create_app():
    app = Flask(__name__)

    app.register_blueprint(NBA_bp)

    # 在 Flask debug 模式下，Werkzeug 会启动两个进程（父进程用于监控改动、子进程用于运行应用）。
    # 通过判断环境变量 `WERKZEUG_RUN_MAIN`，确保只在实际运行的子进程中启动调度器，避免调度器未生效或被父进程退出时终止。
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        nba_scheduler.start()

    @app.after_request
    def add_cors_headers(response):
        response.headers.add("Access-Control-Allow-Origin" , '*')
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True,host='0.0.0.0',port=5002)

    
