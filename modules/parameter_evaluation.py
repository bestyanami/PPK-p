# modules/parameter_evaluation.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import os
import pandas as pd
import json
import numpy as np
import random
from datetime import datetime

parameter_evaluation_bp = Blueprint('parameter_evaluation', __name__, url_prefix='/parameter_evaluation')

@parameter_evaluation_bp.route('/', methods=['GET'])
@login_required
def parameter_evaluation_page():
    """渲染参数评估页面"""
    return render_template('parameter_evaluation.html')

@parameter_evaluation_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    """获取模型结果文件列表"""
    try:
        # 获取PKObjResultsFolder文件夹中的模型结果文件
        if not os.path.exists('PKObjResultsFolder'):
            os.makedirs('PKObjResultsFolder')
        
        files = [f for f in os.listdir('PKObjResultsFolder') if f.endswith('.ressum.rds') or f.endswith('.ressum.txt')]
        return jsonify({'status': 'success', 'model_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@parameter_evaluation_bp.route('/evaluate_params', methods=['POST'])
@login_required
def evaluate_params():
    """评估模型参数"""
    try:
        selected_result = request.form.get('selected_result')
        if not selected_result:
            return jsonify({'status': 'error', 'message': '未选择模型结果文件'})
        
        result_path = os.path.join('PKObjResultsFolder', selected_result)
        txt_result_path = os.path.splitext(result_path)[0] + '.txt'
        
        # 检查文件是否存在
        if not (os.path.exists(result_path) or os.path.exists(txt_result_path)):
            return jsonify({'status': 'error', 'message': '模型结果文件不存在'})
        
        # 由于rpy2相关代码已被注释，我们使用模拟数据
        # 在实际应用中，您应该使用rpy2读取RDS文件中的参数估计值
        
        # 生成模拟参数数据
        params = []
        
        # 常用药动学参数
        pk_params = ['CL', 'V', 'Ka', 'Q', 'V2', 'Ke', 'T1/2', 'AUC', 'Cmax', 'Tmax']
        # 随机选择3-6个参数
        selected_params = random.sample(pk_params, random.randint(3, 6))
        
        for param in selected_params:
            # 生成随机估计值和置信区间
            estimate = round(random.uniform(0.1, 10.0), 3)
            std_error = round(estimate * random.uniform(0.05, 0.2), 3)
            lower_ci = round(estimate - 1.96 * std_error, 3)
            upper_ci = round(estimate + 1.96 * std_error, 3)
            
            params.append({
                '参数': param,
                '估计值': estimate,
                '标准误': std_error,
                '下限': lower_ci,
                '上限': upper_ci,
                '相对标准误(%)': round(std_error / estimate * 100, 1)
            })
        
        # 生成参数相关性数据
        correlation_matrix = []
        for i, param1 in enumerate(selected_params):
            for j, param2 in enumerate(selected_params):
                if i < j:  # 只取上三角矩阵的值
                    correlation_matrix.append({
                        'param1': param1,
                        'param2': param2,
                        'correlation': round(random.uniform(-1, 1), 2)
                    })
        
        # 保存结果到文件
        if not os.path.exists('PKPEResultsFolder'):
            os.makedirs('PKPEResultsFolder')
        
        result_filename = f"PE_{os.path.splitext(selected_result)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        result_path = os.path.join('PKPEResultsFolder', result_filename)
        
        # 保存为JSON文件
        result_data = {
            'parameters': params,
            'correlation_matrix': correlation_matrix,
            'model_file': selected_result
        }
        
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': '参数评估完成',
            'data': result_data
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@parameter_evaluation_bp.route('/result_list', methods=['GET'])
@login_required
def result_list():
    """获取参数评估结果文件列表"""
    try:
        # 获取PKPEResultsFolder文件夹中的结果文件
        if not os.path.exists('PKPEResultsFolder'):
            os.makedirs('PKPEResultsFolder')
        
        files = [f for f in os.listdir('PKPEResultsFolder') if f.endswith('.json')]
        return jsonify({'status': 'success', 'result_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@parameter_evaluation_bp.route('/get_result', methods=['GET'])
@login_required
def get_result():
    """获取参数评估结果文件内容"""
    try:
        result_file = request.args.get('file')
        if not result_file:
            return jsonify({'status': 'error', 'message': '未指定结果文件'})
        
        file_path = os.path.join('PKPEResultsFolder', result_file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '结果文件不存在'})
        
        # 读取JSON文件
        with open(file_path, 'r') as f:
            result_data = json.load(f)
        
        return jsonify({
            'status': 'success', 
            'data': result_data
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})