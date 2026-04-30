"""
配置文件：所有超参数和路径设置
- 集中管理所有超参数
- 提供配置更新接口
- 支持命令行参数覆盖
"""


class Config:
    """全局配置类"""

    # ==================== 数据配置 ====================
    data_root = r'C:\Users\Administrator\Desktop\GMD_vice\new_frame\pressure_vibration1_S2_02'  # 修正为相对于项目根目录的路径
    window_size = 2560
    stride = 1280
    num_classes = 9  # 根据实际数据自动确定
    num_modalities = 2  # 压力 + 振动

    # ==================== 训练配置 ====================
    # 教师模型 使用完整双模态训练
    teacher_epochs = 10
    teacher_lr = 5e-4
    teacher_batch_size = 64

    # NFM生成器 学习P↔V映射
    nfm_epochs = 30
    nfm_lr = 5e-4
    nfm_batch_size = 64

    # 学生模型
    student_epochs = 30
    student_lr = 1e-4
    student_batch_size = 64
    missing_rate = 0.3  # 训练时模态缺失概率  # 训练时模态缺失概率（30% P, 30% V, 40% P+V）

    # ==================== 模型配置 ====================
    # 特征提取器 (1D-CNN) CNN编码器：1D卷积提取时序特征
    cnn_channels = [32, 64, 128] # 通道数逐层增加
    cnn_kernels = [7, 5, 3]# 卷积核逐层减小
    feature_dim = 256  # 修改为256以匹配4层CNN架构 # 输出特征维度

    # Perceiver
    perceiver_latent_dim = 256 # 潜在向量维度
    perceiver_num_latents = 32 # 潜在向量数量
    perceiver_depth = 4 # Perceiver层数
    perceiver_heads = 8  # 注意力头数

    # NFM生成器 频域跨模态映射
    nfm_hidden_dim = 32
    nfm_layer_num = 2
    nfm_lft_hidden = 32
    nfm_dropout = 0.3

    # ==================== 学生模型轻量化配置（方案1：保守改造）====================
    # 这些参数用于轻量化学生模型，减少参数量约58%
    student_encoder_channels = [32]  # 原：[32, 64, 128, 256]，减少1层
    student_latent_dim = 256  # 保持不变
    student_num_latents = 32  # 原：64，减半
    student_perceiver_depth = 3  # 原：4，减少1层
    student_perceiver_heads = 8  # 保持不变
    student_use_dual_perceiver = False  # 原：True，改用单Perceiver
    student_use_transformer_fusion = False  # 原：True，移除Transformer融合

    # ==================== 梯度平衡配置（插件式）====================
    # 🆕 梯度平衡器开关和参数
    use_gradient_balancing = False  # 是否启用梯度平衡（默认关闭，不影响现有代码）
    gradient_balance_alpha = 0.5  # 目标权重：压力模态的期望梯度占比（0.5=平衡，>0.5=偏向压力）
    gradient_balance_momentum = 0.9 # 移动平均系数：平滑梯度统计（0.9=较强平滑）
    gradient_balance_log_interval = 10  # 梯度统计日志打印间隔（每N个batch打印一次）

    # ==================== 损失权重 ====================
    # HMI损失权重
    hmi_alpha = 0.3  # Point-wise MI  （MINE）
    hmi_beta = 0.3# Structural MI （HSIC）
    hmi_gamma = 0.4 # Distributional MI （KL散度）

    # 总损失权重
    lambda_cls = 1.0  # 分类损失
    lambda_hmi = 0.5  # HMI蒸馏损失
    lambda_gmd = 0.3  # GMD正则化损失

    # NFM生成器损失权重
    nfm_lambda_td = 0.4  # 时域损失
    nfm_lambda_fd = 0.5  # 频域损失
    nfm_lambda_cycle = 0.1  # 循环一致性损失

    # ==================== NFM权重衰减配置 ====================
    # 🆕 NFM渐进式权重衰减参数（集中管理）
    nfm_initial_weight = 0.4      # NFM初始权重（训练初期充分利用跨模态补全）
    nfm_final_weight = 0.0        # NFM最终权重（训练后期逐渐适应零填充）
    nfm_warmup_ratio = 0.7      # Warmup阶段比例（前30%保持高权重）
    nfm_decay_ratio = 0.2        # Decay阶段比例（后70%线性衰减）
    nfm_decay_schedule = 'piecewise'  # 衰减策略：'linear', 'cosine', 'piecewise'

    # ==================== 优化器配置 ====================
    weight_decay = 1e-3
    scheduler_patience = 5
    scheduler_factor = 0.5

    # ==================== 路径配置 ====================
    checkpoint_dir = '../checkpoints'
    log_dir = '../logs'

    # 模型保存路径
    teacher_model_path = './checkpoints/teacher_model.pt'
    nfm_generator_path = './checkpoints/nfm_generator.pt'
    student_model_path = './checkpoints/student_model.pt'

    # ==================== 设备配置 ====================
    device = 'cuda'  # 'cuda' or 'cpu'
    num_workers = 0
    seed = 42

    # ==================== 消融实验开关 ====================
    use_nfm_completion = False  # 是否使用NFM补全
    use_hmi_distillation = True  # 是否使用HMI蒸馏
    # use_gmd_decoupling = False  # 是否使用GMD解耦
    
    # ==================== PV-NFM配置 ====================
    # 🆕 PV-NFM模态补全开关（与NFM Generator二选一）
    use_pvnfm = False  # 是否使用PV-NFM替代NFM Generator
    # 🆕 PVNFM架构选择
    use_dual_branch_pvnfm = True  # True=双分支, False=单分支

    # 🆕 循环一致性损失权重
    cycle_consistency_weight = 0.1  # 0.0=禁用, 0.1=推荐值



    # 🆕 纯频域架构开关（仅幅度谱）
    use_magnitude_only = False  # 是否使用纯频域架构（仅幅度谱）
    # 注意：use_magnitude_only=True时，会自动启用use_pvnfm
    
    # PV-NFM模型参数
    pvnfm_context_dim = 128  # 上下文向量维度
    pvnfm_num_bands = 16  # 傅里叶特征频带数
    pvnfm_hidden_inr = 128  # INR隐藏层维度
    
    # ==================== Magnitude-Only架构配置 ====================
    # 🆕 纯频域学生模型参数（student_model_magnitude.py）
    magnitude_F_bins = 1281  # 频域维度（频点数）
    magnitude_num_classes = 9  # 分类类别数
    magnitude_encoder_channels = [32,64]  # CNN编码器通道数
    magnitude_latent_dim = 128  # 潜在特征维度
    magnitude_num_latents = 32  # Perceiver的latent数量
    magnitude_perceiver_depth = 3  # Perceiver深度
    magnitude_perceiver_heads = 8  # Perceiver注意力头数
    magnitude_use_pvnfm = False  # 是否使用PV-NFM
    magnitude_use_dual_perceiver = False  # 是否使用双Perceiver
    magnitude_use_transformer_fusion = False  # 是否使用Transformer融合
    magnitude_decay_schedule = 'piecewise'  # NFM权重衰减策略（None表示使用全局配置）
    magnitude_nfm_initial_weight = 0.3  # NFM初始权重（None表示使用全局配置）
    magnitude_nfm_final_weight = 0.0  # NFM最终权重（None表示使用全局配置）
    magnitude_nfm_warmup_ratio = 0.6  # NFM warmup阶段比例（None表示使用全局配置）
    magnitude_nfm_decay_ratio = 0.2  # NFM decay阶段比例（None表示使用全局配置）
    magnitude_pvnfm_context_dim = 128  # PV-NFM上下文维度
    magnitude_pvnfm_num_bands = 16  # PV-NFM傅里叶频带数
    magnitude_pvnfm_hidden_inr = 128  # PV-NFM INR隐藏层维度

    @classmethod
    def update_from_args(cls, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if hasattr(cls, key):
                setattr(cls, key, value)

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 50)
        print("当前配置:")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)
    
    @classmethod
    def add_magnitude_model_args(cls, parser):
        """
        为命令行参数解析器添加Magnitude-Only模型参数
        用于student_model_magnitude.py的灵活配置
        
        Args:
            parser: argparse.ArgumentParser对象
        
        Returns:
            parser: 添加了magnitude模型参数的parser
        
        使用示例:
            parser = argparse.ArgumentParser()
            Config.add_magnitude_model_args(parser)
            args = parser.parse_args()
        """
        # 创建magnitude模型参数组
        mag_group = parser.add_argument_group('Magnitude-Only Model Parameters',
                                               'Parameters for student_model_magnitude.py')
        
        # 基础架构参数
        mag_group.add_argument('--magnitude_F_bins', type=int, default=cls.magnitude_F_bins,
                              help=f'频域维度（频点数），默认: {cls.magnitude_F_bins}')
        mag_group.add_argument('--magnitude_num_classes', type=int, default=cls.magnitude_num_classes,
                              help=f'分类类别数，默认: {cls.magnitude_num_classes}')
        
        # 编码器参数
        mag_group.add_argument('--magnitude_encoder_channels', type=int, nargs='+',
                              default=cls.magnitude_encoder_channels,
                              help=f'CNN编码器通道数列表，默认: {cls.magnitude_encoder_channels}')
        mag_group.add_argument('--magnitude_latent_dim', type=int, default=cls.magnitude_latent_dim,
                              help=f'潜在特征维度，默认: {cls.magnitude_latent_dim}')
        
        # Perceiver参数
        mag_group.add_argument('--magnitude_num_latents', type=int, default=cls.magnitude_num_latents,
                              help=f'Perceiver的latent数量，默认: {cls.magnitude_num_latents}')
        mag_group.add_argument('--magnitude_perceiver_depth', type=int, default=cls.magnitude_perceiver_depth,
                              help=f'Perceiver深度，默认: {cls.magnitude_perceiver_depth}')
        mag_group.add_argument('--magnitude_perceiver_heads', type=int, default=cls.magnitude_perceiver_heads,
                              help=f'Perceiver注意力头数，默认: {cls.magnitude_perceiver_heads}')
        
        # 架构开关
        mag_group.add_argument('--magnitude_use_pvnfm', action='store_true',
                              default=cls.magnitude_use_pvnfm,
                              help='是否使用PV-NFM模态补全')
        mag_group.add_argument('--magnitude_use_dual_perceiver', action='store_true',
                              default=cls.magnitude_use_dual_perceiver,
                              help='是否使用双Perceiver架构')
        mag_group.add_argument('--magnitude_use_transformer_fusion', action='store_true',
                              default=cls.magnitude_use_transformer_fusion,
                              help='是否使用Transformer融合')
        
        # NFM权重衰减参数
        mag_group.add_argument('--magnitude_decay_schedule', type=str,
                              choices=['linear', 'cosine', 'piecewise'],
                              default=cls.magnitude_decay_schedule,
                              help='NFM权重衰减策略（None表示使用全局配置）')
        mag_group.add_argument('--magnitude_nfm_initial_weight', type=float,
                              default=cls.magnitude_nfm_initial_weight,
                              help='NFM初始权重（None表示使用全局配置）')
        mag_group.add_argument('--magnitude_nfm_final_weight', type=float,
                              default=cls.magnitude_nfm_final_weight,
                              help='NFM最终权重（None表示使用全局配置）')
        mag_group.add_argument('--magnitude_nfm_warmup_ratio', type=float,
                              default=cls.magnitude_nfm_warmup_ratio,
                              help='NFM warmup阶段比例（None表示使用全局配置）')
        mag_group.add_argument('--magnitude_nfm_decay_ratio', type=float,
                              default=cls.magnitude_nfm_decay_ratio,
                              help='NFM decay阶段比例（None表示使用全局配置）')
        
        # PV-NFM参数
        mag_group.add_argument('--magnitude_pvnfm_context_dim', type=int,
                              default=cls.magnitude_pvnfm_context_dim,
                              help=f'PV-NFM上下文维度，默认: {cls.magnitude_pvnfm_context_dim}')
        mag_group.add_argument('--magnitude_pvnfm_num_bands', type=int,
                              default=cls.magnitude_pvnfm_num_bands,
                              help=f'PV-NFM傅里叶频带数，默认: {cls.magnitude_pvnfm_num_bands}')
        mag_group.add_argument('--magnitude_pvnfm_hidden_inr', type=int,
                              default=cls.magnitude_pvnfm_hidden_inr,
                              help=f'PV-NFM INR隐藏层维度，默认: {cls.magnitude_pvnfm_hidden_inr}')

    # ==================== IEB-SDD框架配置 ====================
    # 🆕 信息熵均衡（IEB）配置
    use_ieb = False  # 是否启用IEB频域自适应划分
    ieb_start_epoch = 5  # IEB延迟启动轮次（前N个epoch使用固定k）
    ieb_update_interval = 10  # IEB更新间隔（每N个epoch更新一次）
    ieb_min_k = 7 # 最小频段数量
    ieb_max_k = 7 # 最大频段数量
    ieb_gvf_threshold = 0.7  # GVF阈值（统计显著性）

    # 🆕 尺度解耦蒸馏（SDD）配置
    use_sdd = False  # 是否启用SDD多头蒸馏
    sdd_alpha = 0.7  # 全局-局部损失权重（0.5=平衡）
    sdd_temperature = 2.0  # 蒸馏温度
    sdd_local_weight_strategy = 'uniform'  # 局部头权重策略（'uniform' or 'adaptive'）

    # 🆕 IEB-SDD教师网络配置
    ieb_teacher_encoder_channels = [64, 128, 256]  # 教师编码器通道数
    ieb_teacher_hidden_dim = 256  # 教师隐藏层维度
    ieb_teacher_num_heads = 8  # Transformer注意力头数
    ieb_teacher_num_layers = 4  # Transformer层数
    ieb_teacher_dropout = 0.1  # Dropout率

    # 🆕 IEB-SDD学生网络配置
    ieb_student_encoder_channels = [32, 64]  # 学生编码器通道数（轻量化）
    ieb_student_latent_dim = 128  # 学生特征维度
    ieb_student_num_latents = 32  # Perceiver latent数量
    ieb_student_perceiver_depth = 3  # Perceiver深度
    ieb_student_perceiver_heads = 8  # Perceiver注意力头数
    ieb_student_max_k = 6  # 最大局部头数量

    # 🆕 IEB-SDD损失权重
    ieb_sdd_ce_weight = 1.0  # 分类损失权重
    ieb_sdd_distill_weight = 0.1  # 蒸馏损失权重
    ieb_sdd_recon_weight = 0.01  # 重建损失权重（PV-NFM循环一致性）

    # 🆕 IEB-SDD梯度控制
    ieb_sdd_gradient_clip_norm = 1.0  # 梯度裁剪范数

    # 🆕 消融实验快速切换
    ablation_mode = 'full'  # 'full', 'baseline', 'no_ieb', 'no_sdd', 'no_pvnfm'

    @classmethod
    def get_ablation_config(cls, mode: str):
        """
        根据消融实验模式返回配置

        Args:
            mode: 'full' (完整框架), 'baseline' (无IEB+SDD),
                  'no_ieb' (无IEB), 'no_sdd' (无SDD), 'no_pvnfm' (无PV-NFM)

        Returns:
            dict: 配置字典
        """
        config = {
            'use_ieb': cls.use_ieb,
            'use_sdd': cls.use_sdd,
            'use_pvnfm': cls.use_pvnfm,
            'use_magnitude_only': cls.use_magnitude_only
        }

        if mode == 'full':
            # 完整框架：IEB + SDD + PV-NFM
            config['use_ieb'] = True
            config['use_sdd'] = True
            config['use_pvnfm'] = True
            config['use_magnitude_only'] = True

        elif mode == 'baseline':
            # Baseline：无IEB、无SDD、无PV-NFM
            config['use_ieb'] = False
            config['use_sdd'] = False
            config['use_pvnfm'] = False
            config['use_magnitude_only'] = True

        elif mode == 'no_ieb':
            # 无IEB：固定频域划分
            config['use_ieb'] = False
            config['use_sdd'] = True
            config['use_pvnfm'] = True
            config['use_magnitude_only'] = True

        elif mode == 'no_sdd':
            # 无SDD：仅全局蒸馏
            config['use_ieb'] = True
            config['use_sdd'] = False
            config['use_pvnfm'] = True
            config['use_magnitude_only'] = True

        elif mode == 'no_pvnfm':
            # 无PV-NFM：零填充
            config['use_ieb'] = True
            config['use_sdd'] = True
            config['use_pvnfm'] = False
            config['use_magnitude_only'] = True

        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        return config

    @classmethod
    def add_ieb_sdd_args(cls, parser):
        """
        为命令行参数解析器添加IEB-SDD参数

        Args:
            parser: argparse.ArgumentParser对象

        Returns:
            parser: 添加了IEB-SDD参数的parser
        """
        ieb_sdd_group = parser.add_argument_group('IEB-SDD Framework Parameters')

        # IEB参数
        ieb_sdd_group.add_argument('--use_ieb', action='store_true', default=cls.use_ieb,
                                   help='启用IEB频域自适应划分')
        ieb_sdd_group.add_argument('--ieb_start_epoch', type=int, default=cls.ieb_start_epoch,
                                   help=f'IEB延迟启动轮次，默认: {cls.ieb_start_epoch}')
        ieb_sdd_group.add_argument('--ieb_update_interval', type=int, default=cls.ieb_update_interval,
                                   help=f'IEB更新间隔，默认: {cls.ieb_update_interval}')
        ieb_sdd_group.add_argument('--ieb_min_k', type=int, default=cls.ieb_min_k,
                                   help=f'最小频段数量，默认: {cls.ieb_min_k}')
        ieb_sdd_group.add_argument('--ieb_max_k', type=int, default=cls.ieb_max_k,
                                   help=f'最大频段数量，默认: {cls.ieb_max_k}')
        ieb_sdd_group.add_argument('--ieb_gvf_threshold', type=float, default=cls.ieb_gvf_threshold,
                                   help=f'GVF阈值，默认: {cls.ieb_gvf_threshold}')

        # SDD参数
        ieb_sdd_group.add_argument('--use_sdd', action='store_true', default=cls.use_sdd,
                                   help='启用SDD多头蒸馏')
        ieb_sdd_group.add_argument('--sdd_alpha', type=float, default=cls.sdd_alpha,
                                   help=f'全局-局部损失权重，默认: {cls.sdd_alpha}')
        ieb_sdd_group.add_argument('--sdd_temperature', type=float, default=cls.sdd_temperature,
                                   help=f'蒸馏温度，默认: {cls.sdd_temperature}')
        ieb_sdd_group.add_argument('--sdd_local_weight_strategy', type=str,
                                   choices=['uniform', 'adaptive'],
                                   default=cls.sdd_local_weight_strategy,
                                   help='局部头权重策略')

        # 损失权重
        ieb_sdd_group.add_argument('--ieb_sdd_ce_weight', type=float, default=cls.ieb_sdd_ce_weight,
                                   help=f'分类损失权重，默认: {cls.ieb_sdd_ce_weight}')
        ieb_sdd_group.add_argument('--ieb_sdd_distill_weight', type=float, default=cls.ieb_sdd_distill_weight,
                                   help=f'蒸馏损失权重，默认: {cls.ieb_sdd_distill_weight}')
        ieb_sdd_group.add_argument('--ieb_sdd_recon_weight', type=float, default=cls.ieb_sdd_recon_weight,
                                   help=f'重建损失权重，默认: {cls.ieb_sdd_recon_weight}')

        # 梯度控制
        ieb_sdd_group.add_argument('--ieb_sdd_gradient_clip_norm', type=float,
                                   default=cls.ieb_sdd_gradient_clip_norm,
                                   help=f'梯度裁剪范数，默认: {cls.ieb_sdd_gradient_clip_norm}')

        # 消融实验
        ieb_sdd_group.add_argument('--ablation_mode', type=str,
                                   choices=['full', 'baseline', 'no_ieb', 'no_sdd', 'no_pvnfm'],
                                   default=cls.ablation_mode,
                                   help='消融实验模式')

        return parser



