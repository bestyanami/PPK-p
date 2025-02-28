# modules/data_exploration.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import os
import pandas as pd
import json

data_exploration_bp = Blueprint('data_exploration', __name__, url_prefix='/data_exploration')

@data_exploration_bp.route('/', methods=['GET'])
@login_required
def exploration_page():
    return render_template('exploration.html')
@data_exploration_bp.route('/data_list', methods=['GET'])
@login_required
def data_list():
    try:
        # 获取PKdata文件夹中的RDS文件列表
        if not os.path.exists('PKdata'):
            os.makedirs('PKdata')
        
        files = [f for f in os.listdir('PKdata') if f.endswith('.csv')]
        return jsonify({'status': 'success', 'data_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@data_exploration_bp.route('/load_data', methods=['GET'])
@login_required
def load_data():
    try:
        file_name = request.args.get('file_name')
        if not file_name:
            return jsonify({'status': 'error', 'message': '未提供文件名'})
        
        file_path = os.path.join('PKdata', file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': f'文件 {file_name} 不存在'})
        
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            return jsonify({'status': 'error', 'message': '不支持的文件格式'})
        
        # 准备数据摘要
        summary = df.describe().to_string()
        
        # 准备JSON响应数据
        data = {
            'columns': df.columns.tolist(),
            'data': df.to_dict('records')
        }
        
        return jsonify({
            'status': 'success', 
            'data': data,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})