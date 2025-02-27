# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import json
from datetime import timedelta
import secrets
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# 导入自定义模块
from modules.data_upload import data_upload_bp
from modules.data_exploration import data_exploration_bp
from modules.model_selection import model_selection_bp
from modules.objective_function import objective_function_bp
from modules.covariant_screening import covariant_screening_bp
from modules.parameter_evaluation import parameter_evaluation_bp
from modules.model_diagnosis import model_diagnosis_bp
from modules.dose_recommendation import dose_recommendation_bp

# 初始化应用
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 上传限制

# 初始化 flask-login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 创建必要的文件夹
folders = [
    'PKdata', 'PKModelLibrary', 'PKBaseModelFolder', 'PKObjResultsFolder',
    'PKCovariatesFolder', 'PKPEResultsFolder', 'PKDrawingFolder', 
    'PLModelFolder', 'PLData'
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 简单的用户模型
class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# 注册蓝图
app.register_blueprint(data_upload_bp)
app.register_blueprint(data_exploration_bp)
app.register_blueprint(model_selection_bp)
app.register_blueprint(objective_function_bp)
app.register_blueprint(covariant_screening_bp)
app.register_blueprint(parameter_evaluation_bp)
app.register_blueprint(model_diagnosis_bp)
app.register_blueprint(dose_recommendation_bp)

@app.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 简单认证 - 生产环境中应使用更安全的方法
        if username == "a" and password == "a":  # 与原代码保持一致的简单认证
            user = User("admin")
            login_user(user, remember=True)
            session.permanent = True
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误！')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/help')
@login_required
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)