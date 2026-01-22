from flask_ import NBA_bp
from flask import Flask

def create_app():
    app = Flask(__name__)

    app.register_blueprint(NBA_bp)

    @app.after_request
    def add_cors_headers(response):
        response.headers.add("Access-Control-Allow-Origin" , '*')
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True,host='0.0.0.0',port=5002)

    
