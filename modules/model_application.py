# modules/model_application.py

from flask import Blueprint, render_template, request, jsonify, send_from_directory
from flask_login import login_required
import os
import pandas as pd
import numpy as np
import json
import logging
import joblib
from datetime import datetime

# 创建蓝图
model_application_bp = Blueprint('model_application', __name__, url_prefix='/model_application')

# 确保结果文件夹存在
os.makedirs('Predictions', exist_ok=True)

# 模型应用主页
@model_application_bp.route('/', methods=['GET'])
@login_required
def application_page():
    """渲染模型应用页面"""
    return render_template('model_application.html')

@model_application_bp.route('/analyze_csv', methods=['POST'])
@login_required
def analyze_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '请上传CSV文件'})
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        return jsonify({
            'status': 'success',
            'columns': df.columns.tolist(),
            'rows': len(df),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
# 获取已保存模型列表
@model_application_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    """获取所有可用的模型列表(联邦学习和机器学习)"""
    try:
        models = []
        # 获取联邦学习模型
        if os.path.exists('FL_Models'):
            fl_models = [{'name': f, 'type': '联邦学习'} 
                        for f in os.listdir('FL_Models') if f.endswith('.joblib')]
            models.extend(fl_models)
            
        # 获取机器学习模型
        if os.path.exists('ML_Models'):
            ml_models = [{'name': f, 'type': '机器学习'} 
                        for f in os.listdir('ML_Models') if f.endswith('.joblib')]
            models.extend(ml_models)
            
        return jsonify({'status': 'success', 'models': models})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 加载模型
@model_application_bp.route('/load_model', methods=['POST'])
@login_required
def load_model():
    try:
        model_info = request.json
        model_filename = model_info.get('filename')
        model_type = model_info.get('type')
        
        if not model_filename:
            return jsonify({'status': 'error', 'message': '请选择模型文件'})
        
        # 根据模型类型确定路径
        if model_type == '联邦学习':
            model_folder = 'FL_Models'
        else:
            model_folder = 'ML_Models'
            
        model_path = os.path.join(model_folder, model_filename)
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在'})
            
        # 加载模型
        model = joblib.load(model_path)
        
        # 从元数据文件加载特征信息
        feature_names = []
        metadata_path = os.path.join(model_folder, f"{os.path.splitext(model_filename)[0]}.json")
        metadata = {}
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
                
        # 如果元数据没有特征信息，尝试从模型对象获取
        if not feature_names and hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        
        # 保存到会话
        session_data = {
            'model': model,
            'model_path': model_path,
            'feature_names': feature_names,
            'metadata': metadata
        }
        
        # 将会话数据存储到临时文件
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        session_file = os.path.join('Predictions', f"session_{session_id}.json")
        
        with open(session_file, 'w') as f:
            # 只保存可序列化的信息
            json.dump({
                'session_id': session_id,
                'model_path': model_path,
                'feature_names': feature_names,
                'metadata': {
                    'drug_name': metadata.get('drug_name', '未指定'),
                    'concentration_unit': metadata.get('concentration_unit', '未指定'),
                    'model_type': model_type
                }
            }, f)
        
        return jsonify({
            'status': 'success', 
            'message': '模型加载成功', 
            'session_id': session_id,
            'feature_names': feature_names,
            'drug_name': metadata.get('drug_name', '未指定'),
            'concentration_unit': metadata.get('concentration_unit', '未指定')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 单样本预测
@model_application_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    """使用加载的模型进行单样本预测"""
    try:
        data = request.json
        session_id = data.get('session_id')
        features = data.get('features')
        
        if not session_id or not features:
            return jsonify({'status': 'error', 'message': '缺少必要参数'})
        
        # 从会话文件加载信息
        session_file = os.path.join('Predictions', f"session_{session_id}.json")
        if not os.path.exists(session_file):
            return jsonify({'status': 'error', 'message': '会话已过期，请重新加载模型'})
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # 加载模型
        model_path = session_data['model_path']
        model = joblib.load(model_path)
        
        # 处理输入特征
        input_df = pd.DataFrame([features])
        
        # 预测
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': float(prediction),
            'drug_name': session_data['metadata'].get('drug_name', '未指定'),
            'concentration_unit': session_data['metadata'].get('concentration_unit', '未指定')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 批量预测
@model_application_bp.route('/batch_predict', methods=['POST'])
@login_required
def batch_predict():
    """批量预测功能"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '请上传数据文件'})
        
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'status': 'error', 'message': '请先加载模型'})
            
        # 从会话文件加载信息
        session_file = os.path.join('Predictions', f"session_{session_id}.json")
        if not os.path.exists(session_file):
            return jsonify({'status': 'error', 'message': '会话已过期，请重新加载模型'})
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # 加载模型和特征信息
        model_path = session_data['model_path']
        feature_names = session_data.get('feature_names', [])
        model = joblib.load(model_path)
        
        # 读取上传的CSV文件
        file = request.files['file']
        df = pd.read_csv(file)
        
        # 准备预测数据
        if feature_names and len(feature_names) > 0:
            # 检查缺失特征
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                return jsonify({
                    'status': 'error', 
                    'message': f'输入数据缺少必要特征: {", ".join(missing_features)}'
                })
            
            # 使用已知特征进行预测
            X = df[feature_names].fillna(0)
        else:
            # 默认使用所有数值列
            X = df.select_dtypes(include=[np.number]).fillna(0)
        
        # 执行预测
        predictions = model.predict(X)
        
        # 将预测结果添加到原始数据
        df['predicted_value'] = predictions
        
        # 保存预测结果
        output_filename = f"prediction_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        output_path = os.path.join('Predictions', output_filename)
        df.to_csv(output_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '批量预测完成',
            'output_file': output_filename,
            'rows_processed': len(df)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 文件下载
@model_application_bp.route('/download/<filename>', methods=['GET'])
@login_required
def download_file(filename):
    """下载预测结果文件"""
    try:
        return send_from_directory('Predictions', filename, as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@model_application_bp.route('/prediction_history', methods=['GET'])
@login_required
def prediction_history():
    try:
        if not os.path.exists('Predictions'):
            os.makedirs('Predictions')
            
        files = [f for f in os.listdir('Predictions') 
                if f.startswith('prediction_result_') and f.endswith('.csv')]
        
        history = []
        for f in files:
            try:
                file_path = os.path.join('Predictions', f)
                created_time = datetime.fromtimestamp(os.path.getctime(file_path))
                df = pd.read_csv(file_path)
                
                history.append({
                    'filename': f,
                    'created_at': created_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df)
                })
            except:
                continue
                
        return jsonify({
            'status': 'success',
            'history': history
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})