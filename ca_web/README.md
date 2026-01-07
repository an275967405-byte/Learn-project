# 学生管理系统

一个基于 Flask 和 openpyxl 的简单学生管理系统，数据存储在 Excel 文件中。

## 项目结构

```
an/
├── app.py                      # 应用主入口
├── requirements.txt            # 项目依赖
├── students.xlsx              # 数据文件（自动生成）
├── config/                    # 配置模块
│   ├── __init__.py
│   └── config.py             # 配置文件
├── models/                    # 数据模型
│   ├── __init__.py
│   └── db.py                 # 数据库操作
├── routes/                    # 路由模块
│   ├── __init__.py
│   ├── student_routes.py     # 学生相关路由
│   └── static_routes.py      # 静态文件路由
└── static/                    # 静态资源（如有）
```

## 功能模块说明

### 1. `config/` - 配置模块
- 存放项目配置，如文件路径、Flask配置、CORS配置等
- 便于统一管理和修改配置

### 2. `models/` - 数据模型
- `db.py`: 封装了所有 Excel 数据库操作
- 提供学生的增删改查功能
- 负责数据的持久化和内存管理

### 3. `routes/` - 路由模块
- `student_routes.py`: 学生管理相关的所有API路由
- `static_routes.py`: 静态文件服务路由
- 使用 Blueprint 实现模块化

### 4. `app.py` - 应用入口
- 创建 Flask 应用
- 注册蓝图
- 配置 CORS

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行应用
```bash
python app.py
```

应用将在 `http://127.0.0.1:5000` 启动

## API 接口

### 学生查询
- `GET /student` - 获取所有学生
- `GET /student/query/<name>` - 根据姓名查询
- `GET /student/query/<id>` - 根据ID查询
- `GET /student/level/<name>` - 获取学生等级

### 统计功能
- `GET /student/count/avg` - 计算平均分
- `GET /student/count/max` - 获取最高分
- `GET /student/count/min` - 获取最低分

### 学生管理
- `POST /student` - 添加学生
  ```json
  {
    "name": "张三",
    "score": 85
  }
  ```
- `PUT /student` - 更新学生成绩
  ```json
  {
    "name": "张三",
    "score": 90
  }
  ```
- `DELETE /student` - 删除学生
  ```json
  {
    "name": "张三"
  }
  ```

## 学习要点

1. **模块化设计**: 将单文件拆分为多个模块，职责清晰
2. **Blueprint使用**: 学习Flask蓝图实现路由模块化
3. **配置分离**: 配置与代码分离，便于维护
4. **数据封装**: 使用类封装数据库操作，提供清晰的接口
5. **代码复用**: 避免重复代码，提高可维护性

## 注意事项

- `students.xlsx` 会在首次运行时自动创建
- 数据同时存储在内存和Excel文件中
- 修改配置请编辑 `config/config.py`

