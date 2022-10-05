# math_model_tookits
数学建模代码仓库

# 环 境
pip freeze > requirements.txt 保存环境

pip install -r requirements.txt 部署环境
# 功 能
## 数据预处理 data_preprocess.py
### 数据清洗 def feature_preprocess()：
- **def del_same_feature(data):**
  - 删除方差为0的列（全部相同）
- **def del_perc_same_feature(data, threshold):**
  - 删除相同比列高于阈值的列
- **def del_std_small_feature(data, threshold):**
  - 删除方差小于阈值的列
- **def del_perc_null_feature(data, threshold):**
  - 删除缺失值比例高于阈值的列
    - 可将0替换为缺失值满足实际要求，比如0值占比高于阈值也可以。
- **def sigma3_rules(data):**
  - 利用3σ法则删除特征含有异常值的样本（删除某些行）
- 箱线图处理异常
- **def fill_null(data):**
  - 填充缺失值，支持将0转换成空值处理，方法包含：
    - 前后填充
    - 均值填充
    - 线性插值
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
- **def ShowHeatMap(DataFrame, title):**
  - 画出各种系数热力图
- **def box_plot(df):**
  - 画出模型指标的箱线图
### 特征重要度 def feature_selection():
  - **def grey_top_m(df, target, m=20):**
    - 灰色关联分析
  - **def mic_top_m(df, target, m=20):**
    - 最大信息系数
  - **def dcor_top_m(df, target, m=20):**
    - 距离相关系数
  - **def rf_features(x, y, m=20):**
    - 随机森林重要度和permutation_importance
  - **def feature_selection_graph():**
    - 每个特征的重要度可视化
  - **def feature_integration():**
    - 几个方法的加权集成特征重要度计算，简单加权
    - 实现根据2种相关性自动去除高相关特征并不齐直到没有高相关特征
### 相关性分析 def feature_relation_graph(root):
  - **def grey_all_features(df):**
    - 灰色关联分析
  - **def mic_all_features(df):**
    - 最大信息系数
  - **def dcor_all_features(df):**
    - 距离相关系数
  - **sns.heatmap(features.corr(), square=True, annot=False)**
    - 皮尔逊相关系数
### 模型 regression.py
- 常用模型
  - KNN
  - MLP
  - SVR(SVM)
  - RandomForest
  - XGBoost
- 模型效果比较
  - 见”图表“中的箱线图 **def box_plot(df):**