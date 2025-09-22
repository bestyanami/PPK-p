# PopPK: 集成化的群体药动学分析平台

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&longCache=true&style=for-the-badge)
![R](https://img.shields.io/badge/R-4.2%2B-blue.svg?logo=R&longCache=true&style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-2.x-black.svg?logo=flask&longCache=true&style=for-the-badge)
![nlmixr2](https://img.shields.io/badge/nlmixr2-2.0.8-orange.svg?longCache=true&style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E.svg?logo=scikit-learn&longCache=true&style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**PopPK** 是一个专为药动学研究人员设计的网页版群体药动学（PopPK）建模与分析平台。本研究融合了 Python Flask 的后端服务能力与 R 语言在统计建模方面的强大优势，旨在提供一个从数据探索到模型应用的全流程、可视化、一站式解决方案。

---

## 核心功能

本研究平台的功能设计紧密围绕群体药动学的标准工作流程，并拓展了机器学习与联邦学习等前沿应用。

-   **<i class="fas fa-database"></i> 数据管理与探索**
    -   **数据读入**: 支持用户上传标准格式的 CSV 数据集，并灵活指定 ID, TIME, DV 等关键列。
    -   **数据探索**: 提供交互式的数据表格、摘要统计，并能动态生成个体与群体的药动学曲线图。

-   **<i class="fas fa-cogs"></i> 群体药动学建模**
    -   **基础模型选择**: 内置 `nlmixr2` 语法定义的模型库，用户可选择一室、二室等经典模型进行快速拟合。
    -   **协变量筛选**: 自动化分析年龄、体重等协变量对药动学参数的影响，并以图表形式展示显著性。
    -   **模型评估与诊断**:
        -   **目标函数 (OFV)**: 计算并比较不同模型的 OFV、AIC、BIC 等关键指标。
        -   **参数评估**: 展示模型参数的估计值、标准误和置信区间，并绘制相关性热图。
        -   **模型诊断**: 生成拟合优度图、残差图等多种诊断图表，全面评估模型性能。

-   **<i class="fas fa-brain"></i> 机器学习与应用**
    -   **机器学习**: 集成随机森林等算法，用于预测关键药动学参数。
    -   **联邦学习**: 支持在保护数据隐私的前提下，利用多中心数据协同训练模型。
    -   **模型应用**: 提供统一接口，加载已训练的 PK、ML 或 FL 模型，进行单样本或批量数据的快速预测。
    -   **剂量推荐**: 基于已建立的模型，进行个体化给药方案的模拟与优化。

---

## 系统架构

本研究采用混合架构，以 Python Flask 作为主框架，负责 Web 服务、用户管理和任务调度。核心的群体药动学建模任务通过 `rpy2` 库调用 R 环境中的 `nlmixr2` 包完成，而机器学习功能则由 Python 的 `scikit-learn` 库实现。前端采用 Bootstrap 5 和 Plotly.js 构建，实现了响应式和交互式的用户界面。

---

## 技术栈

-   **后端框架**: Python 3.9+, Flask
-   **前端技术**: HTML5, CSS3, JavaScript, Bootstrap 5, jQuery
-   **核心计算引擎**:
    -   **群体药动学**: R 4.2+, `nlmixr2`, `rxode2`
    -   **Python-R 交互**: `rpy2`
    -   **机器学习**: `scikit-learn`, `joblib`
-   **数据处理**: `pandas`, `numpy`
-   **可视化**: `Plotly.js`
-   **用户管理**: Flask-Login, Werkzeug (密码哈希)

---

## 安装与运行

请遵循以下步骤在本地环境中部署和运行本研究。

### 1. 环境准备

-   **Python**: 确保已安装 Python 3.9 或更高版本。
-   **R**(可选): 确保已安装 R 4.2 或更高版本，并将其添加至系统环境变量。
-   **Rtools** (Windows): 如果您在 Windows 系统上，请安装与 R 版本匹配的 Rtools。

### 2. 克隆与安装依赖

```bash
# 克隆本研究仓库
git clone https://github.com/bestyanami/PPK-p.git
cd ppk-p

# 创建并激活虚拟环境 (推荐)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装 Python 依赖
pip install -r requirements.txt
```

### 3. 安装 R 依赖包

打开 R 控制台，执行以下命令安装必要的 R 包：

```R
# 安装核心建模包
install.packages(c("devtools", "jsonlite", "rpy2"))
devtools::install_github("nlmixr2/nlmixr2")

# 确保其他依赖也已安装
install.packages(c("shiny", "DT", "dplyr", "ggplot2", "rxode2"))
```

### 4. 创建用户

本研究使用基于文件的用户管理系统。运行以下命令创建一个管理员账户：

```bash
# 按照提示输入用户名和密码
python manage_users.py add
```

### 5. 运行应用

```bash
# 启动生产服务器
python run.py
# 若需要启动调试模式
python app.py
```

启动成功后，在浏览器中访问 `http://127.0.0.1:5000` 即可进入登录页面。

---

## 目录结构

```
.
├── app.py                  # Flask 应用主入口
├── modules/                # 各功能模块的蓝图 (Blueprints)
│   ├── data_upload.py
│   ├── model_selection.py
│   ├── machine_learning.py
│   └── ...
├── templates/              # Jinja2 HTML 模板
│   ├── base.html
│   ├── index.html
│   ├── data_upload.html
│   └── ...
├── static/                 # 静态文件 (CSS, JS)
├── PKdata/                 # 存放用户上传的原始数据
├── PKModelLibrary/         # 预定义的 PK 模型库
├── PKObjResultsFolder/     # 存放 nlmixr2 模型运行结果
├── ML_Models/              # 存放训练好的机器学习模型
├── FL_Models/              # 存放训练好的联邦学习模型
├── requirements.txt        # Python 依赖列表
└── manage_users.py         # 用户管理脚本
```

---

## 许可

本研究采用 MIT 许可。详情请见 `LICENSE` 文件。