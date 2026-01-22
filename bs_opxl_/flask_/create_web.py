from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
from bs_data import nba
from date_dir.date_ import data_date_
from excel_dir.create_excel import Excel_


NBA_bp = Blueprint('NBA数据提取', __name__)

# 主页 - 提供HTML页面
@NBA_bp.route('/')
def index():
    return render_template('index.html')

# 静态文件服务
@NBA_bp.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static')
    return send_from_directory(static_dir, filename)

@NBA_bp.route('/extract_data', methods=['POST'])
def extract_data_to_excel():
    print(f"收到请求: {request.method}")
    print(f"请求头: {request.headers}")
    print(f"请求数据: {request.data}")
    try:
        nba.crawling_score()
        nba.crawling_rebound()
        nba.crawling_assist()
        # 检查Excel文件是否成功保存
        import os
        if os.path.exists('NBA.xlsx'):
            return jsonify({'extract data': 'complete!'}), 200
        else:
            return jsonify({'error': 'Excel文件未创建'}), 404
    except Exception as e:
        print(f"数据提取错误: {e}")
        return jsonify({'error': f'数据提取失败: {str(e)}'}), 500

@NBA_bp.route('/show_data', methods=['GET'])
def show_datas():
    datas_ = nba.show_all()
    return jsonify({'NBA_datas': datas_})