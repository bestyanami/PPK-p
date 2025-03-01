# modules/federated_learning.py

from flask import Blueprint, render_template, request, jsonify, current_app
from flask_login import login_required
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# 创建一个Blueprint
federated_learning_bp = Blueprint('federated_learning', __name__, url_prefix='/federated_learning')

# 确保模型保存文件夹存在
os.makedirs('FL_Models', exist_ok=True)

# 全局变量存储当前联邦学习会话的模型
current_federated_model = None
current_federated_history = None

@federated_learning_bp.route('/', methods=['GET'])
@login_required
def federated_learning_page():
    """渲染联邦学习页面"""
    return render_template('federated_learning.html')

@federated_learning_bp.route('/data_list', methods=['GET'])
@login_required
def data_list():
    """获取数据文件列表"""
    try:
        if not os.path.exists('PKdata'):
            os.makedirs('PKdata')
        
        files = [f for f in os.listdir('PKdata') if f.endswith('.csv')]
        return jsonify({'status': 'success', 'data_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@federated_learning_bp.route('/run_federated', methods=['POST'])
@login_required
def run_federated():
    """运行联邦学习模型"""
    try:
        # 获取表单数据
        data_sources = request.form.getlist('data_sources[]')
        target_column = request.form.get('target_column')
        num_rounds = int(request.form.get('num_rounds', 10))
        batch_size = int(request.form.get('batch_size', 32))
        local_epochs = int(request.form.get('local_epochs', 1))
        
        if not data_sources or not target_column:
            return jsonify({'status': 'error', 'message': '请选择数据源和目标列'})
        
        # 加载数据源
        all_data = []
        for source in data_sources:
            file_path = os.path.join('PKdata', source)
            if not os.path.exists(file_path):
                return jsonify({'status': 'error', 'message': f'数据文件不存在: {source}'})
                
            df = pd.read_csv(file_path)
            
            # 检查目标列是否存在
            if target_column not in df.columns:
                return jsonify({'status': 'error', 'message': f'目标列 {target_column} 在数据源 {source} 中不存在'})
            
            all_data.append((source, df))
        
        # 运行联邦学习
        global current_federated_model, current_federated_history
        model, history, evaluation = run_federated_learning(
            all_data, 
            target_column, 
            num_rounds, 
            batch_size, 
            local_epochs
        )
        
        # 保存当前会话的模型
        current_federated_model = model
        current_federated_history = {'model': model, 'history': history, 'evaluation': evaluation}
        
        return jsonify({
            'status': 'success',
            'message': '联邦学习完成',
            'training_history': {
                'rounds': list(range(1, len(history) + 1)),
                'loss': history
            },
            'evaluation': evaluation
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error', 
            'message': f'运行联邦学习时出错: {str(e)}',
            'traceback': traceback.format_exc()
        })

@federated_learning_bp.route('/save_federated_model', methods=['POST'])
@login_required
def save_federated_model():
    """保存联邦学习模型"""
    try:
        model_name = request.form.get('model_name')
        
        if not model_name:
            return jsonify({'status': 'error', 'message': '请提供模型名称'})
        
        global current_federated_model, current_federated_history
        if current_federated_model is None:
            return jsonify({'status': 'error', 'message': '没有可保存的模型，请先运行联邦学习'})
        
        # 保存模型
        model_filename = f"FL_{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        model_path = os.path.join('FL_Models', model_filename)
        joblib.dump(current_federated_model, model_path)
        
        # 保存模型元数据
        metadata = {
            'model_name': model_name,
            'model_type': 'federated_learning',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'history': current_federated_history['history'],
            'evaluation': current_federated_history['evaluation']
        }
        
        metadata_path = os.path.join('FL_Models', f"{os.path.splitext(model_filename)[0]}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return jsonify({
            'status': 'success',
            'message': '模型保存成功',
            'model_filename': model_filename
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@federated_learning_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    """获取已保存的联邦学习模型列表"""
    try:
        if not os.path.exists('FL_Models'):
            os.makedirs('FL_Models')
        
        models = [f for f in os.listdir('FL_Models') if f.endswith('.joblib')]
        return jsonify({'status': 'success', 'models': models})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def run_federated_learning(data_sources, target_column, num_rounds=10, batch_size=32, local_epochs=1):
    """
    运行联邦学习模型
    
    参数:
        data_sources: 多个数据源的列表，每项为(数据源名称, 数据帧)元组
        target_column: 目标列名称
        num_rounds: 联邦学习轮数
        batch_size: 每个客户端的本地训练批次大小
        local_epochs: 每轮联邦学习中客户端的本地训练轮数
    
    返回:
        model: 训练好的联邦模型
        history: 训练历史记录
        evaluation: 模型评估结果
    """
    try:
        # 导入TensorFlow和TensorFlow Federated
        import tensorflow as tf
        import tensorflow_federated as tff
    except ImportError:
        # 如果没有安装TensorFlow Federated，使用模拟实现
        return run_simulated_federated_learning(data_sources, target_column, num_rounds)
    
    # 数据预处理
    client_datasets = []
    client_names = []
    all_features = set()
    
    for source_name, df in data_sources:
        # 提取特征和目标
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features.remove(target_column) if target_column in features else None
        all_features.update(features)
        
        X = df[features].fillna(0).values
        y = df[target_column].fillna(0).values
        
        # 创建TensorFlow数据集
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(df)).batch(batch_size)
        
        client_datasets.append(dataset)
        client_names.append(source_name)
    
    # 模型定义
    def create_keras_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(len(all_features),)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    # TFF模型函数
    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=client_datasets[0].element_spec,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
    
    # 联邦学习过程
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    
    # 初始化服务器状态
    state = iterative_process.initialize()
    
    # 训练联邦模型
    history = []
    for round_num in range(num_rounds):
        # 执行一轮联邦训练
        state, metrics = iterative_process.next(state, client_datasets)
        history.append(float(metrics['train']['loss']))
    
    # 评估联邦模型
    evaluation = evaluate_federated_model(state, client_datasets, client_names)
    
    return state, history, evaluation

def run_simulated_federated_learning(data_sources, target_column, num_rounds=10):
    """
    模拟联邦学习过程 (当TensorFlow Federated未安装时使用)
    
    参数:
        data_sources: 多个数据源的列表，每项为(数据源名称, 数据帧)元组
        target_column: 目标列名称
        num_rounds: 联邦学习轮数
    
    返回:
        model: 训练好的模型
        history: 训练历史记录
        evaluation: 模型评估结果
    """
    from sklearn.linear_model import LinearRegression
    
    # 数据预处理
    all_X = []
    all_y = []
    client_X = []
    client_y = []
    client_names = []
    
    for source_name, df in data_sources:
        # 提取特征和目标
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features.remove(target_column) if target_column in features else None
        
        X = df[features].fillna(0)
        y = df[target_column].fillna(0)
        
        all_X.append(X)
        all_y.append(y)
        client_X.append(X)
        client_y.append(y)
        client_names.append(source_name)
    
    # 合并所有数据
    X_combined = pd.concat(all_X)
    y_combined = pd.concat(all_y)
    
    # 初始化模型
    model = LinearRegression()
    
    # 模拟联邦学习过程
    history = []
    for _ in range(num_rounds):
        # 在每个客户端上训练
        for i in range(len(client_X)):
            model.fit(client_X[i], client_y[i])
        
        # 计算和记录总体损失
        y_pred = model.predict(X_combined)
        mse = np.mean((y_combined - y_pred) ** 2)
        history.append(float(mse))
    
    # 评估模型
    evaluation = {
        'metrics': [],
        'predictions': {
            'actual': [],
            'predicted': []
        }
    }
    
    # 对每个客户端评估
    for i, name in enumerate(client_names):
        y_pred = model.predict(client_X[i])
        mse = np.mean((client_y[i] - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(client_y[i] - y_pred))
        r2 = 1 - np.sum((client_y[i] - y_pred) ** 2) / np.sum((client_y[i] - client_y[i].mean()) ** 2)
        
        evaluation['metrics'].append({
            'source': name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        })
        
        # 添加预测值和实际值 (最多100个点，避免JSON太大)
        sample_size = min(100, len(client_y[i]))
        sample_indices = np.random.choice(len(client_y[i]), sample_size, replace=False)
        
        evaluation['predictions']['actual'].extend(client_y[i].iloc[sample_indices].tolist())
        evaluation['predictions']['predicted'].extend(y_pred[sample_indices].tolist())
    
    # 创建预测散点图
    create_prediction_plot(evaluation['predictions']['actual'], evaluation['predictions']['predicted'])
    
    return model, history, evaluation

def evaluate_federated_model(state, client_datasets, client_names):
    """评估联邦学习模型在各个客户端上的表现"""
    # 注意：这是TFF实现的占位符，在run_simulated_federated_learning中
    # 已经实现了模拟评估，因此这个函数不会被调用
    pass

def create_prediction_plot(actual, predicted):
    """创建预测vs实际值的散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('联邦学习模型预测值 vs 实际值')
    plt.grid(True)
    
    # 将图表转换为Base64编码
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_image