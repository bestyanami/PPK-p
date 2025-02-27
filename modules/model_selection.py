# modules/model_selection.py

from flask import Blueprint, render_template, request, jsonify, current_app, flash
from flask_login import login_required
import os
import pandas as pd
import datetime
import json
import numpy as np

model_selection_bp = Blueprint('model_selection', __name__, url_prefix='/model_selection')

@model_selection_bp.route('/', methods=['GET'])
@login_required
def model_selection_page():
    return render_template('model_selection.html')

@model_selection_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    try:
        # 获取PKModelLibrary文件夹中的模型文件
        if not os.path.exists('PKModelLibrary'):
            os.makedirs('PKModelLibrary')
        
        files = [f for f in os.listdir('PKModelLibrary') if f.endswith('.R')]
        return jsonify({'status': 'success', 'model_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@model_selection_bp.route('/data_list', methods=['GET'])
@login_required
def data_list():
    try:
        # 获取PKdata文件夹中的数据文件
        if not os.path.exists('PKdata'):
            os.makedirs('PKdata')
        
        files = [f for f in os.listdir('PKdata') if f.endswith('.rds') or f.endswith('.csv')]
        return jsonify({'status': 'success', 'data_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@model_selection_bp.route('/run_model', methods=['POST'])
@login_required
def run_model():
    try:
        # 获取表单数据
        selected_model = request.form.get('selected_model')
        selected_data = request.form.get('selected_data')
        
        if not selected_model:
            return jsonify({'status': 'error', 'message': '未选择模型文件'})
        
        if not selected_data:
            return jsonify({'status': 'error', 'message': '未选择数据文件'})
        
        # 检查文件是否存在
        model_path = os.path.join('PKModelLibrary', selected_model)
        data_path = os.path.join('PKdata', selected_data)
        
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在'})
        
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': '数据文件不存在'})
        
        # 以下是模拟模型拟合过程
        # 在实际应用中，您需要使用rpy2调用R的nlmixr函数
        # 或者使用Python的替代库进行模型拟合
        
        try:
            # 模拟模型运行，实际中应使用rpy2调用R代码
            # 假设模型运行成功，生成结果名称
            result_name = f"Result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            result_path = os.path.join('PKObjResultsFolder', f"{result_name}.ressum.rds")
            
            # 创建一个简单的摘要文本
            summary = f"模型：{selected_model}\n数据：{selected_data}\n\n"
            summary += "模拟的模型拟合结果摘要：\n"
            summary += "参数估计：\n"
            summary += "  CL: 1.5 (0.2)\n"
            summary += "  V: 30.0 (4.5)\n"
            summary += "  Ka: 0.8 (0.1)\n"
            summary += "目标函数值: -100.5\n"
            summary += "AIC: 201.0\n"
            summary += "BIC: 215.3\n"
            
            # 在实际应用中，您会将真实的拟合结果保存为RDS文件
            # 这里我们只是模拟这个过程
            
            # 确保结果文件夹存在
            if not os.path.exists('PKObjResultsFolder'):
                os.makedirs('PKObjResultsFolder')
                
            # 模拟保存结果文件（在实际应用中使用rpy2）
            with open(result_path.replace('.rds', '.txt'), 'w') as f:
                f.write(summary)
                
            return jsonify({
                'status': 'success',
                'message': '模型运行成功',
                'summary': summary,
                'result_name': result_name
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'模型运行错误: {str(e)}'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})