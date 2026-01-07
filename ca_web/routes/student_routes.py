"""
学生管理路由模块
"""
from flask import Blueprint, request, jsonify
from models import db

student_bp = Blueprint('student', __name__)


@student_bp.route('/student', methods=["GET"])
def get_students():
    """获取所有学生"""
    students = db.get_all_students()
    return jsonify({"students": students})


@student_bp.route('/student/query/<string:student_name>', methods=['GET'])
def query_name(student_name):
    """根据姓名查询学生"""
    student = db.find_by_name(student_name)
    if student:
        return jsonify({"student": student})
    return jsonify({"error": "can't find the student"}), 404


@student_bp.route('/student/query/<int:student_id>', methods=['GET'])
def query_id(student_id):
    """根据ID查询学生"""
    student = db.find_by_id(student_id)
    if student:
        return jsonify({"student": student})
    return jsonify({"error": "can't find the student"}), 404


@student_bp.route('/student/level/<string:student_name>', methods=['GET'])
def get_level(student_name):
    """获取学生等级"""
    student = db.find_by_name(student_name)
    if not student:
        return jsonify({"error": "can't find the student's score"}), 404
    
    score = student["score"]
    if score >= 90:
        level = "A"
    elif score >= 80:
        level = "B"
    elif score >= 70:
        level = "C"
    elif score >= 60:
        level = "D"
    else:
        level = "F"
    
    return jsonify({"level": level})


@student_bp.route('/student/count/avg', methods=['GET'])
def count_avg():
    """计算平均分"""
    average = db.calculate_average()
    if average is not None:
        return jsonify({"average": average})
    return jsonify({"error": "can't find the students' score"}), 404


@student_bp.route('/student/count/max', methods=['GET'])
def count_max():
    """获取最高分"""
    max_score = db.get_max_score()
    if max_score is not None:
        return jsonify({"max score": max_score})
    return jsonify({"error": "can't find the students' score"}), 404


@student_bp.route('/student/count/min', methods=['GET'])
def count_min():
    """获取最低分"""
    min_score = db.get_min_score()
    if min_score is not None:
        return jsonify({"min score": min_score})
    return jsonify({"error": "can't find the students' score"}), 404


@student_bp.route('/student', methods=["POST"])
def add():
    """添加学生"""
    data = request.get_json()
    if not data or "name" not in data or "score" not in data:
        return jsonify({"error": "can't identify the post"}), 400
    
    new_student = db.add_student(data["name"], data["score"])
    return jsonify({"add complete": new_student["name"]}), 201


@student_bp.route('/student', methods=["PUT"])
def update():
    """更新学生信息"""
    data = request.get_json()
    if not data or "name" not in data or "score" not in data:
        return jsonify({"error": "can't identify the put"}), 400
    
    updated_student = db.update_student(data["name"], data["score"])
    if updated_student:
        return jsonify({"update succeeded": data}), 200
    return jsonify({"error": "can't find the student"}), 404


@student_bp.route('/student', methods=["DELETE"])
def delete():
    """删除学生"""
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "can't identify the DELETE"}), 400
    
    if db.delete_student(data["name"]):
        return jsonify({"delete succeeded": data["name"]}), 200
    return jsonify({"error": "can't find the student"}), 404

