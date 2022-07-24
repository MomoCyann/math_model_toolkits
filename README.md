# math_model_tookits
数学建模代码仓库

# 环境
在终端使用 pip install -r requirements.txt 部署环境

# 现有功能
## data_preprocess.py
### def del_same_feature(data):
删除方差为0的列（全部相同）
### def del_perc_same_feature(data, threshold):
删除相同比列高于阈值的列
### def del_std_small_feature(data, threshold):
删除方差小于阈值的列
### def save_png_to_tiff(name):
保存图表为PNG和TIFF两种格式
### def draw_feature(data):
选择整型、浮点型变量各16个画出分布图<br />
保存图表为PNG和TIFF两种格式<br />
png 1600x1000, tiff-dpi：200 → 2594x1854
### def palette():
调色板
