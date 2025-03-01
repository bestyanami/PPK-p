# modules/machine_learning.py

from flask import Blueprint, render_template, request, jsonify, current_app, send_file
from flask_login import login_required
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib

# 机器学习库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
# 设置非交互式后端
matplotlib.use('Agg')
matplotlib.rcParams['tk.window_focus'] = False
import matplotlib.pyplot as plt
plt.ioff()
import io
import base64


# 创建一个Blueprint
machine_learning_bp = Blueprint('machine_learning', __name__, url_prefix='/machine_learning')

# 确保模型保存文件夹存在
os.makedirs('ML_Models', exist_ok=True)

@machine_learning_bp.route('/', methods=['GET'])
@login_required
def machine_learning_page():
    """渲染机器学习页面"""
    return render_template('machine_learning.html')

@machine_learning_bp.route('/data_list', methods=['GET'])
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

@machine_learning_bp.route('/train_model', methods=['POST'])
@login_required
def train_model():
    """训练机器学习模型"""
    try:
        # 获取表单数据
        selected_data = request.form.get('selected_data')
        target_column = request.form.get('target_column')
        model_type = request.form.get('model_type', 'random_forest')
        test_size = float(request.form.get('test_size', 0.2))
        
        if not selected_data or not target_column:
            return jsonify({'status': 'error', 'message': '请选择数据文件和目标列'})
        
        # 加载数据
        file_path = os.path.join('PKdata', selected_data)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '数据文件不存在'})
            
        df = pd.read_csv(file_path)
        
        # 检查目标列是否存在
        if target_column not in df.columns:
            return jsonify({'status': 'error', 'message': f'目标列 {target_column} 不存在'})
        
        # 准备数据
        # 假设除了目标列之外的所有数值列都是特征
        # 此处可以根据需要进行更复杂的特征工程
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features.remove(target_column) if target_column in features else None
        
        if not features:
            return jsonify({'status': 'error', 'message': '没有找到数值型特征列'})
        
        X = df[features].fillna(0)  # 简单处理缺失值
        y = df[target_column].fillna(0)
        
        # 训练测试集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 根据选择的模型类型训练模型
        model, train_results = train_model_by_type(model_type, X_train, y_train, X_test, y_test)
        
        # 保存模型
        model_name = f"ML_{model_type}_{os.path.splitext(selected_data)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.joblib"
        model_path = os.path.join('ML_Models', model_name)
        joblib.dump(model, model_path)
        
        # 保存模型元数据
        metadata = {
            'model_type': model_type,
            'features': features,
            'target': target_column,
            'data_file': selected_data,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': train_results['metrics'],
            'feature_importance': train_results.get('feature_importance', {})
        }
        
        metadata_path = os.path.join('ML_Models', f"{os.path.splitext(model_name)[0]}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 处理结果中的NaN值，转换为None以便jsonify可以正确处理
        def replace_nan(obj):
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(i) for i in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            return obj
        
        return jsonify({
            'status': 'success',
            'message': '模型训练成功',
            'model_name': model_name,
            'results': replace_nan(train_results),  # 由于jsonify无法处理NaN，需要转换为None
            'features': features
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@machine_learning_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    """获取已训练的模型列表"""
    try:
        if not os.path.exists('ML_Models'):
            os.makedirs('ML_Models')
        
        models = [f for f in os.listdir('ML_Models') if f.endswith('.joblib')]
        return jsonify({'status': 'success', 'models': models})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@machine_learning_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    """使用已训练的模型进行预测"""
    try:
        model_name = request.form.get('model_name')
        input_data = request.form.get('input_data') # JSON格式的输入数据或数据文件
        
        if not model_name:
            return jsonify({'status': 'error', 'message': '请选择模型'})
            
        # 加载模型
        model_path = os.path.join('ML_Models', model_name)
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在'})
            
        model = joblib.load(model_path)
        
        # 加载模型元数据
        metadata_path = os.path.join('ML_Models', f"{os.path.splitext(model_name)[0]}.json")
        if not os.path.exists(metadata_path):
            return jsonify({'status': 'error', 'message': '模型元数据不存在'})
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 解析输入数据
        # 这里需要更完善的逻辑来处理不同形式的输入
        if input_data:
            try:
                # 尝试解析JSON输入
                input_df = pd.DataFrame(json.loads(input_data))
            except:
                # 可能是文件名
                input_path = os.path.join('PKdata', input_data)
                if os.path.exists(input_path):
                    input_df = pd.read_csv(input_path)
                else:
                    return jsonify({'status': 'error', 'message': '无法解析输入数据'})
        else:
            return jsonify({'status': 'error', 'message': '请提供输入数据'})
        
        # 确保所需特征都存在
        missing_features = [f for f in metadata['features'] if f not in input_df.columns]
        if missing_features:
            return jsonify({
                'status': 'error', 
                'message': f'输入数据缺少以下特征: {", ".join(missing_features)}'
            })
        
        # 准备预测数据
        X_pred = input_df[metadata['features']].fillna(0)
        
        # 进行预测
        predictions = model.predict(X_pred)
        
        # 将预测结果添加到原数据中
        input_df['predicted_' + metadata['target']] = predictions
        
        # 保存带预测结果的数据
        result_filename = f"pred_{os.path.splitext(model_name)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        result_path = os.path.join('ML_Models', result_filename)
        input_df.to_csv(result_path, index=False)
        
        return jsonify({
            'status': 'success',
            'message': '预测完成',
            'result_file': result_filename,
            'predictions': predictions.tolist(),
            'data_preview': input_df.head(5).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@machine_learning_bp.route('/model_details', methods=['GET'])
@login_required
def model_details():
    """获取模型详细信息"""
    try:
        model_name = request.args.get('model_name')
        if not model_name:
            return jsonify({'status': 'error', 'message': '请指定模型名称'})
            
        # 加载模型元数据
        metadata_path = os.path.join('ML_Models', f"{os.path.splitext(model_name)[0]}.json")
        if not os.path.exists(metadata_path):
            return jsonify({'status': 'error', 'message': '模型元数据不存在'})
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'metadata': metadata
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@machine_learning_bp.route('/download_result', methods=['GET'])
@login_required
def download_result():
    """下载预测结果文件"""
    try:
        file_name = request.args.get('file')
        if not file_name:
            return jsonify({'status': 'error', 'message': '未指定文件名'})
            
        file_path = os.path.join('ML_Models', file_name)
        if not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': '文件不存在'})
        
        return send_file(file_path, 
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name=file_name)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# 配置matplotlib支持中文显示
def configure_matplotlib_chinese():
    """配置matplotlib以支持中文字符"""
    import matplotlib.font_manager as fm
    # 尝试找到支持中文的字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    found_font = None
    
    # 获取系统字体列表
    font_paths = fm.findSystemFonts()
    for font_path in font_paths:
        try:
            font = fm.FontProperties(fname=font_path)
            if font.get_name() in chinese_fonts:
                found_font = font_path
                break
        except:
            continue
    
    if found_font:
        # 设置matplotlib字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=found_font).get_name()]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    else:
        # 如果没有找到中文字体，使用英文替代中文标题
        print("Warning: No Chinese font found, using English titles instead.")

def train_model_by_type(model_type, X_train, y_train, X_test, y_test):
    """根据模型类型训练不同的机器学习模型"""
    # 配置中文字体支持
    configure_matplotlib_chinese()
    results = {}
    
    if model_type == 'random_forest':
        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 模型评估
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # 交叉验证
        if len(y_train) >= 10:  # 确保有足够的样本进行交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(y_train)), scoring='r2')
            cv_r2_mean = round(float(np.nanmean(cv_scores)), 4)
            cv_r2_std = round(float(np.nanstd(cv_scores)), 4)
        else:
            # 样本太少，不进行交叉验证
            cv_scores = np.array([np.nan])
            cv_r2_mean = np.nan
            cv_r2_std = np.nan
        
        # 生成特征重要性数据
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))

        # 检测ID列是否为最重要特征
        id_warning = None
        top_feature = max(feature_importance.items(), key=lambda x: x[1])
        if top_feature[0].upper() == 'ID' or 'id' in top_feature[0].lower():
            id_warning = f"警告：检测到ID列 ({top_feature[0]}) 是最重要的特征，重要性为 {top_feature[1]:.4f}。这可能表明模型过拟合。建议在特征中排除ID列。"
        
        # 生成预测vs实际值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('随机森林模型预测值 vs 实际值')
        plt.grid(True)
        
        # 将图表转换为Base64编码
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 生成特征重要性条形图
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        importances.plot(kind='bar')
        plt.title('特征重要性')
        plt.grid(True)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        importance_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # 整理结果
        results = {
            'metrics': {
                'train_r2': round(float(train_r2), 4),
                'test_r2': round(float(test_r2), 4),
                'train_rmse': round(float(train_rmse), 4),
                'test_rmse': round(float(test_rmse), 4),
                'cv_r2_mean': cv_r2_mean,
                'cv_r2_std': cv_r2_std
            },
            'feature_importance': feature_importance,
            'plots': {
                'prediction_scatter': plot_image,
                'importance_plot': importance_image
            },
            'id_warning': id_warning
        }
        
    # 这里可以添加更多的模型类型
    # elif model_type == 'xgboost':
    # elif model_type == 'neural_network':
    # 等等
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model, results

# 这里可以添加TensorFlow Federated相关功能
# def setup_federated_learning():
#     import tensorflow as tf
#     import tensorflow_federated as tff
#     # 实现联邦学习相关功能