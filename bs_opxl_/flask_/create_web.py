from flask import Blueprint, request, jsonify, render_template, send_from_directory
import os
from bs_data import nba
from date_dir.date_ import data_date_
from excel_dir.create_excel import Excel_, EXCEL_FILE_PATH
from scheduler_dir.scheduler_extract import nba_scheduler
from datetime import datetime


NBA_bp = Blueprint('NBA数据提取', __name__)

# 主页 - 提供HTML页面
@NBA_bp.route('/')
def index():
    return render_template('index.html')

# 静态文件服务
@NBA_bp.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(os.path.dirname(__file__), '..','..', 'static')
    return send_from_directory(static_dir, filename)

@NBA_bp.route('/extract_data', methods=['POST'])
def extract_data_to_excel():
    print(f"收到请求: {request.method}")
    print(f"请求头: {request.headers}")
    print(f"请求数据: {request.data}")
    try:
        # 执行数据爬取
        score_success = nba.crawling_score()
        rebound_success = nba.crawling_rebound()
        assist_success = nba.crawling_assist()
        
        # 检查是否所有爬取都成功
        if score_success and rebound_success and assist_success:
            # 检查Excel文件是否成功保存
            if os.path.exists(EXCEL_FILE_PATH):
                return jsonify({
                    'extract data': 'complete!',
                    'message': '数据爬取成功并已保存到Excel文件',
                    'file_path': EXCEL_FILE_PATH
                }), 200
            else:
                return jsonify({'error': 'Excel文件未创建'}), 404
        else:
            return jsonify({
                'error': '数据爬取部分失败',
                'score_success': score_success,
                'rebound_success': rebound_success,
                'assist_success': assist_success
            }), 500
    except Exception as e:
        print(f"数据提取错误: {e}")
        return jsonify({'error': f'数据提取失败: {str(e)}'}), 500

@NBA_bp.route('/show_data', methods=['GET'])
def show_datas():
    try:
        datas_ = nba.show_all()
        return jsonify({'NBA_datas': datas_})
    except Exception as e:
        print(f"显示数据时出错: {e}")
        return jsonify({'error': f'获取数据失败: {str(e)}'}), 500

@NBA_bp.route('/top_players', methods=['GET'])
def get_top_players():
    try:
        all_data = nba.show_all()
        
        # 修复：data_date_是列表，不是函数
        current_date = data_date_[0] if data_date_ and len(data_date_) > 0 else "未知日期"
        
        top_players = {
            'top_scorer': {'name': "暂无数据", 'score': "0"},
            'top_rebounder': {'name': "暂无数据", 'rebounds': "0"},
            'top_assister': {'name': "暂无数据", 'assists': "0"},
            'date': current_date
        }
        
        # 查找当天数据（第一行是表头，第二行开始是数据）
        for row in all_data[1:]:  # 跳过表头
            if row[0] == '得分' and str(row[1]) == '1':  # 排名第一
                top_players['top_scorer']['name'] = row[2] if row[2] else "未知"
                try:
                    top_players['top_scorer']['score'] = float(row[3]) if row[3] else 0
                except:
                    top_players['top_scorer']['score'] = 0
                
                # 篮板数据（第6-9列）
            if row[5] == '篮板' and str(row[6]) == '1':  # 排名第一
                top_players['top_rebounder']['name'] = row[7] if row[7] else "未知"
                try:
                    top_players['top_rebounder']['rebounds'] = float(row[8]) if row[8] else 0
                except:
                    top_players['top_rebounder']['rebounds'] = 0
                
                # 助攻数据（第11-14列）
            if row[10] == '助攻' and str(row[11]) == '1':  # 排名第一
                top_players['top_assister']['name'] = row[12] if row[12] else "未知"
                try:
                    top_players['top_assister']['assists'] = float(row[13]) if row[13] else 0
                except:
                    top_players['top_assister']['assists'] = 0
        
        # 确保函数总是返回一个响应
        return jsonify(top_players), 200

    except Exception as e:
        print(f'获取最高数据球员出错: {e}')
        return jsonify({'error': f'获取最高数据球员出错: {str(e)}'}), 500


@NBA_bp.route('/scheduler/status', methods=['GET'])
def get_scheduler_status():
    """获取定时任务状态"""
    try:
        jobs = nba_scheduler.scheduler.get_jobs()
        job_info = []
        
        for job in jobs:
            job_info.append({
                'id': job.id,
                'name': job.name,
                'trigger': str(job.trigger),
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'running': False  # APScheduler没有直接的running属性，这里设为False
            })
        
        # 获取最后爬取时间（需要从调度器或日志中获取）
        # 这里暂时使用当前时间作为示例
        last_crawl_time = datetime.now().isoformat()
        
        return jsonify({
            'scheduler_running': nba_scheduler.scheduler.running,
            'status': 'running' if nba_scheduler.scheduler.running else 'stopped',
            'last_crawl_time': last_crawl_time,
            'jobs': job_info,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'获取定时任务状态失败: {str(e)}'}), 500

@NBA_bp.route('/scheduler/trigger', methods=['POST'])
def trigger_crawl_manually():
    """手动触发爬取任务"""
    try:
        success = nba_scheduler.crawl_nba_data()
        if success:
            return jsonify({'message': '手动爬取任务执行成功'}), 200
        else:
            return jsonify({'error': '手动爬取任务执行失败'}), 500
    except Exception as e:
        return jsonify({'error': f'手动触发爬取失败: {str(e)}'}), 500

@NBA_bp.route('/scheduler/start', methods=['POST'])
def start_scheduler():
    """启动定时任务"""
    try:
        nba_scheduler.start()
        return jsonify({'message': '定时任务已启动'}), 200
    except Exception as e:
        return jsonify({'error': f'启动定时任务失败: {str(e)}'}), 500

@NBA_bp.route('/scheduler/stop', methods=['POST'])
def stop_scheduler():
    """停止定时任务"""
    try:
        nba_scheduler.shutdown()
        return jsonify({'message': '定时任务已停止'}), 200
    except Exception as e:
        return jsonify({'error': f'停止定时任务失败: {str(e)}'}), 500

