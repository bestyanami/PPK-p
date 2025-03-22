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

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('federated_learning.log'), logging.StreamHandler()]
)

# 创建一个Blueprint
federated_learning_bp = Blueprint('federated_learning', __name__, url_prefix='/federated_learning')

# 确保模型保存文件夹存在
os.makedirs('FL_Models', exist_ok=True)
# 添加这行以确保预测结果文件夹存在
os.makedirs('Predictions', exist_ok=True)

# 全局变量存储当前联邦学习会话的模型
current_federated_model = None
current_federated_history = None
privacy_metrics = None  # 隐私度量存储

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
    try:
        # 获取表单数据
        data_sources = request.form.getlist('data_sources[]')
        target_column = request.form.get('target_column')
        num_rounds = int(request.form.get('num_rounds', 10))
        batch_size = int(request.form.get('batch_size', 32))
        local_epochs = int(request.form.get('local_epochs', 1))
        
        # 隐私参数
        noise_multiplier = float(request.form.get('noise_multiplier', 0.1))
        l2_norm_clip = float(request.form.get('l2_norm_clip', 1.0))
        secure_aggregation = request.form.get('secure_aggregation', 'true').lower() == 'true'

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
        global current_federated_model, current_federated_history, privacy_metrics
        model, history, evaluation = run_federated_learning(
            all_data, 
            target_column, 
            num_rounds, 
            batch_size, 
            local_epochs,
            noise_multiplier,
            l2_norm_clip,
            secure_aggregation
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
            'evaluation': evaluation,
            'privacy_metrics': evaluation.get('privacy_metrics', {})
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"联邦学习错误: {str(e)}\n{error_details}")
        # 添加错误可视化功能
        return jsonify({
            'status': 'error', 
            'message': f'运行联邦学习时出错: {str(e)}',
            'traceback': error_details,
            'error_type': e.__class__.__name__,
        })

@federated_learning_bp.route('/save_federated_model', methods=['POST'])
@login_required
def save_federated_model():
    try:
        model_name = request.form.get('model_name')
        drug_name = request.form.get('drug_name', '未指定')  # 添加药物名称
        concentration_unit = request.form.get('concentration_unit', '未指定')  # 添加浓度单位
        
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
            'drug_name': drug_name,  # 新增
            'concentration_unit': concentration_unit,  # 新增
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'history': current_federated_history['history'],
            'evaluation': current_federated_history['evaluation'],
            'feature_names': current_federated_history.get('feature_names', [])
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

def run_federated_learning(data_sources, target_column, num_rounds=10, batch_size=32, 
                          local_epochs=1, noise_multiplier=0.1, l2_norm_clip=1.0,
                          secure_aggregation=True):
    """
    运行具有隐私保护的联邦学习模型
    
    参数:
        data_sources: 多个数据源的列表，每项为(数据源名称, 数据帧)元组
        target_column: 目标列名称
        num_rounds: 联邦学习轮数
        batch_size: 每个客户端的本地训练批次大小
        local_epochs: 每轮联邦学习中客户端的本地训练轮数
        noise_multiplier: 差分隐私噪声乘数
        l2_norm_clip: 梯度裁剪阈值
        secure_aggregation: 是否使用安全聚合
    
    返回:
        model: 训练好的联邦模型
        history: 训练历史记录
        evaluation: 模型评估结果
    """
    
    def create_dp_keras_model():
        """创建带差分隐私保护的模型"""
        try:
            from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
        except ImportError:
            # 如果没有安装privacy库，返回普通模型
            return create_keras_model()
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(len(all_features),),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # 使用差分隐私优化器
        optimizer = dp_optimizer_keras.DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            learning_rate=0.1
        )
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    try:
        # 导入TensorFlow和TensorFlow Federated
        import tensorflow as tf
        import tensorflow_federated as tff
        logging.info("使用TensorFlow Federated进行联邦学习")
    except ImportError:
        # 如果没有安装TensorFlow Federated，使用模拟实现
        logging.warning("TensorFlow Federated未安装，使用模拟实现替代")
        return run_simulated_federated_learning(data_sources, target_column, num_rounds)
    
    # 数据预处理
    client_datasets = []
    client_names = []
    all_features = set()
    
    # 计算差分隐私保证
    try:
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
        eps, delta = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=sum(len(df) for _, df in data_sources),
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=num_rounds,
            delta=1e-5
        )
        global privacy_metrics
        privacy_metrics = {
            "epsilon": float(eps),
            "delta": float(delta),
            "noise_multiplier": noise_multiplier,
            "secure_aggregation": secure_aggregation
        }
    except ImportError:
        privacy_metrics = {
            "note": "隐私度量不可用 - 未安装tensorflow_privacy",
            "secure_aggregation": secure_aggregation
        }

    # 数据预处理
    for source_name, df in data_sources:
        # 提取特征和目标，明确忽略ID列
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        # 忽略ID和其他非特征列
        pk_ignore_columns = ['ID', 'USUBJID', 'STUDYID', 'MDV', 'EVID', 'CMT']
        for col in pk_ignore_columns:
            if col in features:
                features.remove(col)
                logging.info(f"在联邦学习中忽略列: {col}")
                
        if target_column in features:
            features.remove(target_column)
        
        # 智能缺失值处理
        X_df = df[features].copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].fillna(X_df[col].median())
            # 标准化数值特征
            if X_df[col].std() > 0:
                X_df[col] = (X_df[col] - X_df[col].mean()) / X_df[col].std()
        
        X = X_df.values
        y = df[target_column].fillna(df[target_column].median()).values
        
        # 创建TensorFlow数据集
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(df)).batch(batch_size)
        # 添加数据集到联邦学习客户端列表
        client_datasets.append(dataset)
        client_names.append(source_name)
    
    # 模型定义
    def create_keras_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(len(all_features),)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
    
    # 修改模型函数
    def model_fn():
        """使用差分隐私模型"""
        keras_model = create_dp_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=client_datasets[0].element_spec,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
    
    # 联邦学习过程
    if secure_aggregation:
        try:
            secure_aggregator = tff.aggregators.SecureAggregator()
            logging.info("启用安全聚合机制")
            iterative_process = tff.learning.build_federated_averaging_process(
                model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
                model_update_aggregation_factory=secure_aggregator
            )
        except Exception as e:
            error_msg = f"安全聚合初始化失败: {str(e)}，使用标准聚合"
            logging.warning(error_msg)
            # 在前端显示警告
            privacy_metrics["secure_aggregation_error"] = error_msg
            iterative_process = tff.learning.build_federated_averaging_process(
                model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
            )
    else:
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

    # 添加缺失的评估和返回部分
    # 评估联邦模型
    evaluation = evaluate_federated_model(state, client_datasets, client_names, create_keras_model)

    # 从服务器状态中提取模型
    keras_model = create_keras_model()
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)

    # 返回结果
    return keras_model, history, evaluation

def run_simulated_federated_learning(data_sources, target_column, num_rounds=10, batch_size=32):
    """模拟联邦学习过程 (当TensorFlow Federated未安装时使用)"""
    from sklearn.linear_model import LinearRegression
    
    # 数据预处理
    all_X = []
    all_y = []
    client_X = []
    client_y = []
    client_names = []
    all_features = set()
    
    # 数据预处理实现
    for source_name, df in data_sources:
        # 提取特征和目标
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        # 忽略ID和其他非特征列
        pk_ignore_columns = ['ID', 'USUBJID', 'STUDYID', 'MDV', 'EVID', 'CMT']
        for col in pk_ignore_columns:
            if col in features:
                features.remove(col)
        if target_column in features:
            features.remove(target_column)
        all_features.update(features)
        
        # 智能缺失值处理
        X_df = df[features].copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].fillna(X_df[col].median())
            # 标准化数值特征
            if X_df[col].std() > 0:
                X_df[col] = (X_df[col] - X_df[col].mean()) / X_df[col].std()
        
        # 准备客户端数据
        client_X.append(X_df)
        client_y.append(df[target_column].fillna(df[target_column].median()))
        client_names.append(source_name)
        
        # 收集所有数据用于全局评估
        all_X.append(X_df)
        all_y.append(df[target_column].fillna(df[target_column].median()))
    
    # 合并所有数据 - 修复缩进(移出循环)
    X_combined = pd.concat(all_X)
    y_combined = pd.concat(all_y)
    
    # 初始化模型 - 修复缩进(移出循环)
    model = LinearRegression()
    
    # 修正模拟联邦学习过程 - 修复缩进(移出循环)
    history = []
    for _ in range(num_rounds):
        # 在每个客户端上训练
        for i in range(len(client_X)):
            model.fit(client_X[i], client_y[i])
        
        # 计算和记录总体损失
        y_pred = model.predict(X_combined)
        mse = np.mean((y_combined - y_pred) ** 2)
        history.append(float(mse))
    
    # 评估模型 - 修复缩进(移出循环)
    evaluation = {
        'metrics': [],
        'predictions': {
            'actual': [],
            'predicted': []
        }
    }
    # 对每个客户端评估 - 保持正确缩进
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
        
        # 添加预测值和实际值
        sample_size = min(100, len(client_y[i]))
        sample_indices = np.random.choice(len(client_y[i]), sample_size, replace=False)
        
        evaluation['predictions']['actual'].extend(client_y[i].iloc[sample_indices].tolist())
        evaluation['predictions']['predicted'].extend(y_pred[sample_indices].tolist())
    
    # 创建预测散点图 - 修复缩进(移出循环)
    plot_image = create_prediction_plot(
        evaluation['predictions']['actual'],
        evaluation['predictions']['predicted']
    )
    evaluation['plot'] = plot_image
    
    # 返回结果 - 修复缩进(移出循环)
    return model, history, evaluation

# 实现评估函数
def evaluate_federated_model(state, client_datasets, client_names, model_fn):
    """评估联邦学习模型在各个客户端上的表现"""
    import tensorflow as tf
    
    # 使用传入的模型函数
    keras_model = model_fn()
    
    # 从联邦状态获取权重并应用
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)
    
    evaluation = {
        'metrics': [],
        'predictions': {'actual': [], 'predicted': []},
        'privacy_metrics': privacy_metrics
    }
    
    # 评估每个客户端
    for i, (dataset, name) in enumerate(zip(client_datasets, client_names)):
        x_all, y_all = [], []
        
        # 收集数据用于评估
        for x, y in dataset:
            x_all.append(x.numpy())
            y_all.append(y.numpy())
            
        if not x_all:  # 防止空数据集
            continue
            
        x_test = np.vstack(x_all)
        y_test = np.concatenate(y_all)
        
        # 在本地预测 - 无需向服务器发送数据
        y_pred = keras_model.predict(x_test).flatten()
        
        # 计算指标
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - mse / np.var(y_test) if np.var(y_test) > 0 else 0
        
        evaluation['metrics'].append({
            'source': name,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        })
        
        # 保存部分结果用于可视化
        sample_size = min(100, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        evaluation['predictions']['actual'].extend(y_test[indices].tolist())
        evaluation['predictions']['predicted'].extend(y_pred[indices].tolist())
    
    # 生成预测散点图
    plot_image = create_prediction_plot(
        evaluation['predictions']['actual'],
        evaluation['predictions']['predicted']
    )
    evaluation['plot'] = plot_image
    
    return evaluation

def create_prediction_plot(actual, predicted):
    """创建预测vs实际值的散点图"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('联邦学习模型预测值 vs 实际值')
    plt.grid(True)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_image

@federated_learning_bp.route('/load_model', methods=['POST'])
@login_required
def load_model():
    try:
        model_filename = request.form.get('model_filename')
        if not model_filename:
            return jsonify({'status': 'error', 'message': '请选择模型文件'})
            
        model_path = os.path.join('FL_Models', model_filename)
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在'})
            
        # 加载模型与元数据
        global current_federated_model, current_federated_history
        current_federated_model = joblib.load(model_path)
        
        # 加载元数据和特征信息
        feature_names = []
        metadata_path = os.path.join('FL_Models', f"{os.path.splitext(model_filename)[0]}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                current_federated_history = {
                    'model': current_federated_model,
                    'history': metadata.get('history', []),
                    'evaluation': metadata.get('evaluation', {})
                }
                feature_names = metadata.get('feature_names', [])
                
        # 特征名缺失时提取可能的特征信息
        if not feature_names and hasattr(current_federated_model, 'feature_names_in_'):
            feature_names = current_federated_model.feature_names_in_.tolist()
        
        return jsonify({
            'status': 'success', 
            'message': '模型加载成功', 
            'model_features': feature_names,
            'drug_name': metadata.get('drug_name', '未指定'),
            'concentration_unit': metadata.get('concentration_unit', '未指定')
        })
    except Exception as e:
        logging.error(f"模型加载错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})
    
@federated_learning_bp.route('/analyze_csv', methods=['POST'])
@login_required
def analyze_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '请上传CSV文件'})
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # 提取列信息
        columns = df.columns.tolist()
        
        # 分析字段类型
        column_types = {}
        numeric_columns = []
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                column_types[col] = 'numeric'
                if col not in ['ID', 'USUBJID', 'STUDYID', 'MDV', 'EVID', 'CMT']:
                    numeric_columns.append(col)
            else:
                column_types[col] = 'categorical'
        
        return jsonify({
            'status': 'success',
            'columns': columns,
            'numeric_columns': numeric_columns,
            'column_types': column_types,
            'sample_data': df.head(3).to_dict('records')
        })
    except Exception as e:
        logging.error(f"CSV分析错误: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})