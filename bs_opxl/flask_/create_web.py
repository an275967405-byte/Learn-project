from flask import Blueprint,request,jsonify
from bs_data import nba
from date_dir.date_ import data_date_
from excel_dir.create_excel import Excel_


NBA_bp = Blueprint('NBA数据提取',__name__)

@NBA_bp.route('/extract_data',methods = ['POST'])
def extract_data_to_excel():
    print(f"收到请求: {request.method}")
    print(f"请求头: {request.headers}")
    print(f"请求数据: {request.data}")
    nba.crawling_score()
    nba.crawling_rebound()
    nba.crawling_assist()
    if Excel_[data_date_[0]]:
        return jsonify({'extract data':'complete!'}),200
    else:
        return jsonify({'error':'extract wrong'}),404

# @NBA_bp.route('/extract_data',methods = ['GET'])
# def show_datas():
#     datas_ = nba.show_all()
#     return jsonify({'NBA datas:':datas_})