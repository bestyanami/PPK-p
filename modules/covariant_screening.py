# modules/covariant_screening.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import os
import pandas as pd
import json
import numpy as np
from scipy import stats
import re

covariant_screening_bp = Blueprint('covariant_screening', __name__, url_prefix='/covariant_screening')

@covariant_screening_bp.route('/', methods=['GET'])
@login_required
def screening_page():
    return render_template('covariant_screening.html')

@covariant_screening_bp.route('/data_list', methods=['GET'])
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

@covariant_screening_bp.route('/get_covariates', methods=['POST'])
@login_required
def get_covariates():
    try:
        selected_data = request.form.get('selected_data')
        if not selected_data:
            return jsonify({'status': 'error', 'message': '未选择数据文件'})
        
        file_path = os.path.join('PKdata', selected_data)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '数据文件不存在'})
        
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.rds'):
            # 由于rpy2注释掉了，我们处理不了RDS文件
            return jsonify({'status': 'error', 'message': '暂不支持RDS格式，请上传CSV文件'})
        
        # 假设协变量包括除了 ID、TIME、DV、AMT 之外的所有列
        non_covariates = ['ID', 'TIME', 'DV', 'AMT']
        covariates = [col for col in df.columns if col not in non_covariates]
        
        return jsonify({'status': 'success', 'covariates': covariates})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@covariant_screening_bp.route('/run_screening', methods=['POST'])
@login_required
def run_screening():
    try:
        selected_data = request.form.get('selected_data')
        selected_covariates = request.form.getlist('selected_covariates[]')
        
        if not selected_data:
            return jsonify({'status': 'error', 'message': '未选择数据文件'})
        
        if not selected_covariates:
            return jsonify({'status': 'error', 'message': '未选择协变量'})
        
        file_path = os.path.join('PKdata', selected_data)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '数据文件不存在'})
        
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.rds'):
            return jsonify({'status': 'error', 'message': '暂不支持RDS格式，请上传CSV文件'})
        
        # 运行协变量筛选（单变量线性回归分析）
        results = []
        for covariate in selected_covariates:
            if covariate in df.columns:
                # 排除缺失值
                df_subset = df.dropna(subset=[covariate, 'DV'])
                
                # 验证数据类型
                try:
                    x = df_subset[covariate].astype(float)
                    y = df_subset['DV'].astype(float)
                    
                    # 线性回归
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    results.append({
                        '协变量': covariate,
                        'p值': p_value,
                        '截距': intercept,
                        '截距_p值': 0 if len(df_subset) < 3 else stats.ttest_1samp(y - slope * x, 0).pvalue,
                        '斜率': slope,
                        '斜率_p值': p_value,
                        '显著性': '显著' if p_value < 0.05 else '不显著',
                        'R平方': r_value ** 2
                    })
                except Exception as e:
                    results.append({
                        '协变量': covariate,
                        'p值': None,
                        '截距': None,
                        '截距_p值': None,
                        '斜率': None,
                        '斜率_p值': None,
                        '显著性': '错误',
                        'R平方': None,
                        '错误': str(e)
                    })
        
        # 保存结果到PKCovariatesFolder
        if not os.path.exists('PKCovariatesFolder'):
            os.makedirs('PKCovariatesFolder')
            
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 生成结果文件名
        result_filename = f"cov_screen_{os.path.splitext(selected_data)[0]}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
        result_path = os.path.join('PKCovariatesFolder', result_filename)
        
        # 保存结果
        results_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success', 
            'message': '协变量筛选完成',
            'results': results,
            'result_file': result_filename
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 获取已有结果文件列表
@covariant_screening_bp.route('/result_list', methods=['GET'])
@login_required
def result_list():
    try:
        if not os.path.exists('PKCovariatesFolder'):
            os.makedirs('PKCovariatesFolder')
        
        files = [f for f in os.listdir('PKCovariatesFolder') if f.endswith('.csv')]
        return jsonify({'status': 'success', 'result_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 获取结果文件内容
@covariant_screening_bp.route('/get_result', methods=['GET'])
@login_required
def get_result():
    try:
        result_file = request.args.get('file')
        if not result_file:
            return jsonify({'status': 'error', 'message': '未指定结果文件'})
        
        file_path = os.path.join('PKCovariatesFolder', result_file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '结果文件不存在'})
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        results = df.to_dict('records')
        
        return jsonify({
            'status': 'success', 
            'results': results
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})