"""
静态文件路由模块
"""
import os
from flask import Blueprint, send_file, jsonify
from config import BASE_DIR

static_bp = Blueprint('static', __name__)


@static_bp.route('/')
def serve_index():
    """提供首页HTML文件"""
    file_path = os.path.join(BASE_DIR, "student_management.html")
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "index file not found"}), 404


@static_bp.route('/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    # 防止访问API路由
    if filename == "student" or filename.startswith('student/'):
        return jsonify({"error": "non-compliant"}), 404
    
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "not the file"}), 404

