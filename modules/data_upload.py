# modules/data_upload.py

from flask import Blueprint, render_template, request, jsonify, current_app, flash
from flask_login import login_required
from werkzeug.utils import secure_filename
import os
import pandas as pd
import shutil
#from rpy2 import robjects
#from rpy2.robjects import pandas2ri
import re

data_upload_bp = Blueprint('data_upload', __name__, url_prefix='/data_upload')

@data_upload_bp.route('/', methods=['GET'])
@login_required
def data_upload_page():
    return render_template('data_upload.html')

@data_upload_bp.route('/upload', methods=['POST'])
@login_required
def upload_data():
    if 'data_file' not in request.files:
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    file = request.files['data_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        upload_path = os.path.join('PKdata', filename)
        
        # 确保目录存在
        os.makedirs('PKdata', exist_ok=True)
        
        # 保存上传的文件到PKdata目录
        file.save(upload_path)
        
        try:
            # 获取表单数据
            col_id = request.form.get('col_id', 'ID')
            col_time = request.form.get('col_time', 'TIME')
            col_dv = request.form.get('col_dv', 'DV')
            col_amt = request.form.get('col_amt', 'AMT')
            
            # 读取CSV文件
            df = pd.read_csv(upload_path)
            
            # 重命名列
            columns_map = {
                col_id: 'ID',
                col_time: 'TIME',
                col_dv: 'DV',
                col_amt: 'AMT'
            }
            
            # 只重命名存在的列
            for old_name, new_name in columns_map.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
            
            # 确保PKdata目录存在
            os.makedirs('PKdata', exist_ok=True)
            
            # 保存处理后的数据为CSV文件到PKdata文件夹
            pk_csv_path = os.path.join('PKdata', filename)
            df.to_csv(pk_csv_path, index=False)
            
            # 如果需要RDS格式（当rpy2可用时）
            # RDS部分被注释掉了
            #pandas2ri.activate()
            #r_df = pandas2ri.py2rpy(df)
            #output_filename = os.path.splitext(filename)[0] + ".rds"
            #output_path = os.path.join('PKdata', output_filename)
            #robjects.r(f'saveRDS(r_df, file="{output_path}")')
            
            return jsonify({
                'status': 'success', 
                'message': '数据上传成功！文件已保存到PKdata文件夹。',
                'filename': filename
            })
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'处理出错: {str(e)}'})
        
    return jsonify({'status': 'error', 'message': '不支持的文件格式，请上传CSV文件'})

@data_upload_bp.route('/file_list', methods=['GET'])
@login_required
def file_list():
    """获取PKdata文件夹中的文件列表"""
    try:
        if not os.path.exists('PKdata'):
            os.makedirs('PKdata')
        
        files = [f for f in os.listdir('PKdata') if f.endswith('.csv') or f.endswith('.rds')]
        return jsonify({'status': 'success', 'files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})