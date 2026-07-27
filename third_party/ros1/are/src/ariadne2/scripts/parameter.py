CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 2.0 # meter

FREE = 0
OCCUPIED = 100
UNKNOWN = -1

SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.5 * SENSOR_RANGE  # for each node, frontiers in this range will be considered as utility
MIN_UTILITY = 3  # if the number observable frontiers is less than this value, consider it is zero utility
FRONTIER_CELL_SIZE = 1 * CELL_SIZE  # downsample the frontiers based on this value

UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # the minimal map that contains all possible updating nodes

NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 25

THR_TO_WAYPOINT = 1 # meter, the waypoint will be considered as arrived if the robot is closer than this value
THR_NEXT_WAYPOINT = 5 # meter, the planner will try to plan a waypoint farther than this value
THR_GRAPH_HARD_UPDATE = 8 # meter, node and edges in this range will be fully updated

CLUSTER_RANGE = 10 # meter, frontiers will be clustered based on this range

AVOID_OSCILLATION = True # if the planner outputs back and forth waypoints, move to one of them
ENABLE_SAVE_MODE = False # if the planner outputs waypoints in loop, move to the nearest frontier
ENABLE_DSTARLITE = False # Use D*-lite for graph rarefaction instead of A*

# RL策略选择模式
GREEDY_ACTION_SELECTION = False  # 是否使用贪婪策略选择（True=更平滑但可能陷入局部最优，False=随机采样更具探索性）

# ============== 新增性能优化参数 ==============
# 碰撞检测优化
ALLOW_UNKNOWN_PASSTHROUGH = False  # 是否允许穿过未知区域（针对狭窄通道优化）
UNKNOWN_TOLERANCE_CELLS = 2  # 允许穿过的最大连续未知格子数量
SOFT_COLLISION_CHECK = False  # 软碰撞检测模式：只有OCCUPIED才算碰撞

# 邻居节点发现优化
EXTENDED_NEIGHBOR_RANGE = False  # 是否使用扩展的邻居搜索范围（7x7而非5x5）
NEIGHBOR_MATRIX_SIZE = 5  # 邻居矩阵大小（5或7）

# 图连通性优化
DELAYED_NODE_REMOVAL = False  # 延迟删除断开连接的节点
NODE_REMOVAL_DELAY_STEPS = 3  # 延迟删除的步数

# 前沿点处理优化
DISTANCE_WEIGHTED_UTILITY = False  # 是否使用距离加权的效用计算
UTILITY_DISTANCE_DECAY = 0.1  # 距离衰减系数

# D*-Lite优化
DSTARLITE_MAX_TIME = 0.1  # D*-Lite最大搜索时间（秒）
DSTARLITE_ADAPTIVE_TIME = False  # 是否自适应调整搜索时间

# 狭窄通道检测
ENABLE_NARROW_PASSAGE_DETECTION = False  # 是否启用狭窄通道检测
NARROW_PASSAGE_WIDTH_THRESHOLD = 3.0  # 狭窄通道宽度阈值（米）

# 图连通性检查优化
USE_CURRENT_LOCATION_FOR_CONNECTIVITY = False  # 使用当前位置而非起点检查连通性

# 小波自适应图阈值（仅影响图稀疏化阈值，不改网络输入维度）
WAVELET_ADAPTIVE_DTH = False  # 是否启用小波自适应距离阈值
WAVELET_DTH_ALPHA = 1.0  # 阈值放大系数（响应低时放宽）
WAVELET_DTH_MAX_MULT = 2.0  # 阈值最大放大倍数
WAVELET_DTH_SCALE_MULTS = (1, 2, 4)  # 相对基础尺度（NODE_RESOLUTION/CELL_SIZE）的倍数
WAVELET_LOCAL_MAP_SIZE = 56.0  # 小波局部地图边长（米）；仅在该窗口内计算小波图
WAVELET_CACHE_CHANGE_RATIO_THRESH = 0.01  # 局部地图变化比例低于该值时复用缓存

# 邻居轻筛选（仅影响候选动作集，不改网络输入维度）
WAVELET_NEIGHBOR_LIGHT_FILTER = False  # 是否启用邻居轻筛选
WAVELET_NEIGHBOR_MAX_KEEP = 12  # 邻居最大保留数（超过时裁剪）
WAVELET_NEIGHBOR_MIN_KEEP = 6  # 邻居最少保留数
WAVELET_NEIGHBOR_FILTER_MIN_TRIGGER = 12  # 候选邻居数超过该值才触发轻筛选
WAVELET_NEIGHBOR_SCORE_WAVELET_WEIGHT = 0.55  # 小波响应权重
WAVELET_NEIGHBOR_SCORE_UTILITY_WEIGHT = 0.30  # utility权重
WAVELET_NEIGHBOR_SCORE_DISTANCE_WEIGHT = 0.15  # 距离权重（更近得分更高）
