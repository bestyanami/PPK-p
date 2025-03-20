# modules/model_selection.py

import logging
import os
import pandas as pd
import datetime
import json
import numpy as np
import subprocess
import tempfile
import contextlib
from flask import Blueprint, render_template, request, jsonify, current_app, flash
from flask_login import login_required
from rpy2 import robjects
from rpy2.robjects import pandas2ri, numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# 创建日志记录器
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def r_inline():
    """确保R代码执行时有正确的转换上下文"""
    with localconverter(default_converter) as cv:
        yield    

model_selection_bp = Blueprint('model_selection', __name__, url_prefix='/model_selection')

@model_selection_bp.route('/', methods=['GET'])
@login_required
def model_selection_page():
    return render_template('model_selection.html')

@model_selection_bp.route('/model_list', methods=['GET'])
@login_required
def model_list():
    try:
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
        if not os.path.exists('PKdata'):
            os.makedirs('PKdata')
        
        files = [f for f in os.listdir('PKdata') if f.endswith('.rds')]
        return jsonify({'status': 'success', 'data_files': files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@model_selection_bp.route('/run_model', methods=['POST'])
@login_required
def run_model():
    try:
        selected_model = request.form.get('selected_model')
        selected_data = request.form.get('selected_data')
        
        if not selected_model:
            return jsonify({'status': 'error', 'message': '未选择模型文件'})
        if not selected_data:
            return jsonify({'status': 'error', 'message': '未选择数据文件'})
        
        model_path = os.path.join('PKModelLibrary', selected_model).replace('\\', '/')
        data_path = os.path.join('PKdata', selected_data).replace('\\', '/')
        
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': '模型文件不存在'})
        if not os.path.exists(data_path):
            return jsonify({'status': 'error', 'message': '数据文件不存在'})
        
        try:
            # 激活数据类型转换器
            numpy2ri.activate()
            pandas2ri.activate()
            
            result_name = f"Result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            result_path = os.path.join('PKObjResultsFolder', f"{result_name}.ressum.rds").replace('\\', '/')
            result_json = os.path.join('PKObjResultsFolder', f"{result_name}_result.json").replace('\\', '/')
            
            if not os.path.exists('PKObjResultsFolder'):
                os.makedirs('PKObjResultsFolder')
            
            # 创建临时R脚本时明确指定UTF-8编码
            r_script = tempfile.NamedTemporaryFile(suffix=".R", mode="w", encoding='utf-8', delete=False)
            r_script.write(f'''
            # 加载必要库
            library(nlmixr2)
            library(rxode2)
            library(jsonlite)

            # 设置文件路径
            model_path <- "{model_path}"
            data_path <- "{data_path}"
            result_path <- "{result_path}"
            result_json <- "{result_json}"

            # 读取数据(从CSV转换为RDS格式)
            data <- read.csv(data_path)

            # 加载模型定义并验证结构
            source(model_path)
            print(ls()) # 输出加载的对象

            # 检查模型对象
            if (exists("mod") && !exists("model")) {{
            model <- mod  # 重命名为model
            cat("Model renamed from mod to model\\n")
            print(body(model))
            }}

            # 验证模型结构
            if (!exists("model")) {{
            write_json(list(status="error", message="Model object not found"), result_json, auto_unbox=TRUE)
            quit(status=1)
            }}

            # 验证模型结构包含model()块
            model_body <- deparse(body(model))
            if (!any(grepl("model\\\\(", model_body))) {{
            write_json(list(status="error", message="Model function missing required model() block"), 
                        result_json, auto_unbox=TRUE)
            quit(status=1)
            }}
            ''')
            r_script.close()
            
            # 执行R脚本并记录时间
            logger.info(f"开始拟合模型: {selected_model} 使用数据: {selected_data}")
            start_time = datetime.datetime.now()

            # 设置执行环境变量，确保编码一致性
            env = os.environ.copy()
            env['LC_ALL'] = 'en_US.UTF-8'
            env['LANG'] = 'en_US.UTF-8'
            env['PYTHONIOENCODING'] = 'utf-8'

            # 使用环境变量执行R脚本
            result = subprocess.run(
                ["Rscript", "--vanilla", "--encoding=UTF-8", r_script.name],
                capture_output=True,
                text=False,
                env=env
            )

            # 增强的错误处理和解码逻辑
            try:
                stdout = result.stdout.decode('utf-8', errors='replace')
                stderr = result.stderr.decode('utf-8', errors='replace')
            except Exception as decode_error:
                logger.error(f"输出解码错误: {str(decode_error)}")
                stdout = str(result.stdout)
                stderr = str(result.stderr)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            logger.info(f"R脚本执行完成，耗时: {elapsed_time:.2f}秒")

            # 检查R脚本执行状态
            if result.returncode != 0:
                logger.error(f"R脚本执行失败，返回码: {result.returncode}")
                logger.error(f"标准错误: {stderr}")
                
                debug_script = os.path.join('PKObjResultsFolder', f"{result_name}_debug.R")
                with open(debug_script, 'w', encoding='utf-8') as f:
                    with open(r_script.name, 'r', encoding='utf-8') as src:
                        f.write(src.read())
                        
                os.unlink(r_script.name)
                return jsonify({
                    'status': 'error', 
                    'message': f'R脚本执行失败: {stderr[:300]}...'
                })

            os.unlink(r_script.name)
            
            # 处理执行结果
            if os.path.exists(result_json):
                with open(result_json, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                status = result_data.get('status')
                if status == 'error':
                    return jsonify({'status': 'error', 'message': result_data.get('message')})
                
                return jsonify({
                    'status': 'success',
                    'message': '模型拟合成功',
                    'summary': result_data.get('summary', ''),
                    'metrics': result_data.get('metrics', {}),
                    'result_name': result_name,
                    'elapsed_time': elapsed_time
                })
            else:
                return jsonify({'status': 'error', 'message': 'R程序运行失败，未生成结果文件'})
                
        except Exception as inner_e:
            logger.error(f"R脚本执行过程错误: {str(inner_e)}")
            return jsonify({'status': 'error', 'message': f'R脚本执行错误: {str(inner_e)}'})
    
    except Exception as e:
        logger.error(f"模型运行错误: {str(e)}")
        return jsonify({'status': 'error', 'message': f'模型运行错误: {str(e)}'})