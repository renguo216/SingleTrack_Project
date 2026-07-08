import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIAMFC_DIR = r'D:\SiamFC_RPN\SiamFC'
SIAMRPNPP_DIR = r'D:\SiamFC_RPN\SiamRPNpp'
TRADITIONAL_DIR = r'D:\SiamFC_RPN\traditional\src'

# SiamFC 模型
SIAMFC_MODEL_PATH = r'D:\SiamFC_RPN\SiamFC\snapshot_mytrain\siamfc_alexnet_e30.pth'

# SiamRPN++ 模型
SIAMRPNPP_CONFIG = r'D:\SiamFC_RPN\SiamRPNpp\experiments\siamrpn_r50_l234_dwxcorr\config.yaml'
SIAMRPNPP_MODEL = r'D:\SiamFC_RPN\SiamRPNpp\experiments\siamrpn_r50_l234_dwxcorr\model.pth'

# 上传和结果目录
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 允许的视频格式
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB

# 设备
DEVICE = 'cuda'  # 或 'cpu'