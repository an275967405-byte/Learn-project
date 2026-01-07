"""
配置文件
"""
import os

# 基础目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Excel文件配置
EXCEL_FILENAME = "students.xlsx"
EXCEL_PATH = os.path.join(BASE_DIR, EXCEL_FILENAME)

# Flask配置
DEBUG = True
HOST = '0.0.0.0'
PORT = 5001

# CORS配置
CORS_ORIGIN = '*'
CORS_HEADERS = 'Content-Type,Authorization'
CORS_METHODS = 'GET,PUT,POST,DELETE,OPTIONS'

