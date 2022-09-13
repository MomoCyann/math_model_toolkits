# math_model_tookits
数学建模代码仓库

# 环 境
pip freeze > requirements.txt 保存环境

pip install -r requirements.txt 部署环境
# 功 能
## 数据清洗 data_preprocess.py
- **def del_same_feature(data):**
  - 删除方差为0的列（全部相同）
- **def del_perc_same_feature(data, threshold):**
  - 删除相同比列高于阈值的列
- **def del_std_small_feature(data, threshold):**
  - 删除方差小于阈值的列
- **def del_perc_null_feature(data, threshold):**
  - 删除缺失值比例高于阈值的列
    - 可将0替换为缺失值满足实际要求
- **def sigma3_rules(data):**
  - 利用3σ法则删除特征含有异常值的样本（删除某些行）
- 箱线图处理异常
- **def fill_null(data):**
  - 填充缺失值，支持将0转换成空值处理，方法包含：
    - 前后填充
    - 均值填充
    - 线性插值
---
## 特征工程 feature_engineering.py
- 相关性分析
- 特征重要度
  - **def grey_relation_analysis(DataFrame):**
    - 灰色关联分析
  - **def mic(df):**
    - 最大信息系数
- 降维
### 图表
- **def save_png_to_tiff(name):**
  - 保存图表为PNG和TIFF两种格式
- **def draw_feature(data):**
  - 选择整型、浮点型变量各16个画出分布图
    - 保存图表为PNG和TIFF两种格式
    - png 1600x1000, tiff-dpi：200 → 2594x1854
    - 可手动指定变量名
    - 可添加子图总标题，默认无
- **def palette():**
  - 调色板，设置配色
    - 渐变色
    - 蓝红简约渐变
    - 彩虹
    - 自定义
### 3.模型
- 常用模型
- 模型效果比较