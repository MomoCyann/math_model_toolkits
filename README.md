# math_model_tookits
数学建模代码仓库

# 环境
pip freeze > requirements.txt 保存环境

pip install -r requirements.txt 部署环境
# 现有功能
## 数据预处理 data_preprocess.py
### 1.特征工程 
- def del_same_feature(data):删除方差为0的列（全部相同）
- def del_perc_same_feature(data, threshold):删除相同比列高于阈值的列
- def del_std_small_feature(data, threshold):删除方差小于阈值的列
- def del_perc_null_feature(data, threshold):删除缺失值比例高于阈值的列
  - 可将0替换为缺失值满足实际要求
- def save_png_to_tiff(name):保存图表为PNG和TIFF两种格式
- def draw_feature(data):选择整型、浮点型变量各16个画出分布图<br>
  - 保存图表为PNG和TIFF两种格式<br>
  - png 1600x1000, tiff-dpi：200 → 2594x1854<br>
  - 可手动指定变量名
  - 可添加子图总标题，默认无
- def palette():调色板
### 2.异常值处理、缺失值处理
- def sigma3_rules(data):利用3σ法则删除特征含有异常值的样本（删除某些行）
- 