# modules/dose_recommendation.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import os
import pandas as pd
import json
import numpy as np

dose_recommendation_bp = Blueprint('dose_recommendation', __name__, url_prefix='/dose_recommendation')

@dose_recommendation_bp.route('/', methods=['GET'])
@login_required
def recommendation_page():
    return render_template('recommendation.html')

@dose_recommendation_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    try:
        # 获取PKObjResultsFolder文件夹中的模型结果文件
        if not os.path.exists('PKObjResultsFolder'):
            os.makedirs('PKObjResultsFolder')
        
        files = [f for f in os.listdir('PKObjResultsFolder') if f.endswith('.ressum.rds')]
        return jsonify({'status': 'success', 'model_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@dose_recommendation_bp.route('/calculate', methods=['POST'])
@login_required
def calculate_dose():
    try:
        # 获取表单数据
        data = request.form
        selected_model = data.get('selected_model')
        desired_dv = float(data.get('desired_dv', 10))
        loading_dose = float(data.get('loading_dose', 100))
        maintenance_dose = float(data.get('maintenance_dose', 50))
        interval = float(data.get('interval', 12))
        
        # 简单剂量调整计算
        # 在实际应用中，你应该使用真实的PK模型
        recommended_dose = maintenance_dose * (desired_dv / (loading_dose * 0.12))
        recommended_dose = round(recommended_dose, 2)
        
        # 创建模拟数据用于绘图
        time = np.arange(0, 72, 0.5)
        simulated_concentration = []
        
        for t in time:
            dose_points = int(t / interval)
            conc = loading_dose * np.exp(-0.1 * (t % interval))
            
            for i in range(dose_points):
                conc += maintenance_dose * np.exp(-0.1 * ((t - interval * (i + 1)) % interval))
            
            simulated_concentration.append(conc)
        
        # 创建响应数据
        result = {
            'loading_dose': loading_dose,
            'maintenance_dose': recommended_dose,
            'interval': interval,
            'desired_dv': desired_dv,
            'simulated_dv': round(np.mean(simulated_concentration), 2)
        }
        
        # 创建绘图数据
        plot_data = {
            'traces': [
                {
                    'x': time.tolist(),
                    'y': simulated_concentration,
                    'mode': 'lines',
                    'name': '模拟浓度',
                    'line': {'color': 'rgb(55, 128, 191)', 'width': 3}
                },
                {
                    'x': [time[0], time[-1]],
                    'y': [desired_dv, desired_dv],
                    'mode': 'lines',
                    'name': '期望浓度',
                    'line': {'color': 'rgb(219, 64, 82)', 'width': 2, 'dash': 'dash'}
                }
            ],
            'layout': {
                'title': '剂量推荐模拟结果',
                'xaxis': {'title': '时间 (小时)'},
                'yaxis': {'title': '药物浓度 (DV)'}
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': result,
            'plot_data': plot_data
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})