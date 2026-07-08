# SingleTrack_Project-单目标跟踪方法与智能分析系统
  本项目是一个基于 **传统数字图像处理** 与 **深度学习** 的单目标跟踪实验与演示平台。项目包含四个核心模块：传统跟踪算法、SiamFC 跟踪器、SiamRPN++ 跟踪器以及一个可供交互的 Web 可视化平台

##项目结构
 本项目通过模块化，实现了算法、实验与Web展示的完全解耦

```text
SingleTrack_Project/
├── traditional/                    # 【模块1】传统视觉跟踪方法
│   ├── data/                       # 测试数据集（GOT-10k）
│   ├── src/                        # 核心源码
│   ├── scripts/                    # 运行脚本
│   └── results/                    # 输出结果
│
├── SiamFC/                         # 【模块2】SiamFC 深度学习跟踪器
│   ├── SiamFC/                     # 网络定义、损失函数、数据处理
│   ├── tools/                      # 训练与测试脚本
│   ├── pretrained/                 # 预训练骨干网络权重
│   ├── snapshot_*/                 # 训练过程保存的模型快照
│   └── results/                    # 测试结果与可视化
│
├── SiamRPNpp/                      # 【模块3】SiamRPN++ 深度学习跟踪器
│   ├── pysot/                      # 核心代码库
│   ├── experiments/                # 实验配置文件
│   ├── pretrained_models/          # 预训练权重
│   ├── testing_dataset/            # 测试数据集（OTB100）
│   └── results/                    # 跟踪结果与对比分析
│
├── tracking_web/                   # 【模块4】Web 可视化交互平台
│   ├── app.py                      # Flask 后端启动入口
│   ├── config.py                   # 配置文件
│   ├── wrappers/                   # 算法调用封装层
│   ├── static/                     # 前端静态资源（CSS/JS）
│   └── otb_videos/                 # 预处理后的测试视频
│
├── tools/                          # 公共工具脚本
├── requirements.txt                # 项目总依赖库
└── README.md                       # 项目说明文档

##运行环境与依赖
 操作系统：Windows 11
 编程语言：Python 3.8+
 深度学习框架：PyTorch（搭配CUDA）

##依赖安装
 pip install -r requirements.txt

##项目数据集使用GOT-10k与OTB100，数据集

##参考项目如下
 https://github.com/forschumi/SiamTrackers
 https://github.com/STVIR/pysot

