# Unitree Go1

## Go1 with Isaac Gym Flow

Official Repo：https://github.com/unitreerobotics/unitree_rl_gym/tree/main

### configure files

`g1_config.py`

+ `G1RoughCfg`

  `LeggedRobotCfg`的子类，用于配置环境 & 机器人静态参数

  + `init_state` 初始状态

    + **`pos = [0,0,0.8]`**
       将机器人出生高度从默认（1.0 m）降低到 0.8 m，以便在更“崎岖”地形上更快接触地面。

    + **`default_joint_angles`**
       覆盖了 12 个关节在“action=0”／“不动”时的位置命令。相比基类中的 0 rad，这里对髋关节抬高 (-0.1 rad)、膝关节弯曲 (0.3 rad)、踝关节略微倾斜 (-0.2 rad)，让机器人进入一个略带弯曲的站立姿态，以更适合在粗糙地形上行走。

  + `env` 环境维度

    + **`num_observations = 47`**
       本任务中，最终的观测向量长度为 47。一会儿结合 `G1Robot` 的 `compute_observations()` 会更清楚。
    + **`num_privileged_obs = 50`**
       非对称训练（Asymmetric Actor-Critic）下提供给 Critic 的“特权观测”长度为 50。通常比普通观测多出速度或位置信息。
    + **`num_actions = 12`**
       与机器人 12 个自由度（DoF）对应，每步策略输出 12 维动作。

  + `domain_rand`

    + **`randomize_friction = True` & `friction_range = [0.1,1.25]`**
       针对地面摩擦系数在 [0.1, 1.25] 范围内随机采样，增强调节对摩擦变化的鲁棒性。

    - **`randomize_base_mass = True` & `added_mass_range = [-1.,3.]`**

       随机地给机器人底盘加负重/轻量：在 [-1 kg, +3 kg] 范围内随机扰动，增加对质量变化的鲁棒性。

    + **`push_robots = True` & `push_interval_s = 5` & `max_push_vel_xy = 1.5`**
       每隔 5 秒对机器人施加一次“外力冲击”，让其瞬间获得最多 ±1.5 m/s 的水平速度。这会让机器人学习在受到外界扰动时保持平衡。

    + `control`

      + **`control_type = 'P'`**
         使用位置控制（PD 控制缺省为 P 模式 + 在 `LeggedRobot` 内部还会加阻尼项）。

      + **`stiffness = {...}` 与 `damping = {...}`**
         关节刚度和阻尼分别针对髋关节、膝关节、踝关节做了调节：

        - 髋关节 (`hip_yaw/hip_roll/hip_pitch`) 刚度 100 N·m/rad、阻尼 2 N·m·s/rad

        - 膝关节 (`knee`) 刚度 150 N·m/rad、阻尼 4 N·m·s/rad

        - 踝关节 (`ankle`) 刚度 40 N·m/rad、阻尼 2 N·m·s/rad
           这些值要比基类更大，确保能在崎岖地形上快速收敛到目标位置。

      - **`action_scale = 0.25`**

        实际施加到关节的目标位置 = `default_joint_angle + 0.25 × action_value`，把动作映射到相对较小的角度增量，以避免关节一次“抖动”过大。

      - **`decimation = 4`**

        策略每输出一次动作，在物理仿真中会执行 4 个控制周期。若仿真主步长 `sim_params.dt=0.005 s`，那么每个策略步相当于 0.02 s。

    + `asset`

      + **`file = '…/g1_12dof.urdf'`**
         指定加载一个 G1 机器狗的 12 DoF URDF 文件。

      + **`name = "g1"`**
         给所有创建的 Actor 一个统一的名字 `g1`。

      + **`foot_name = "ankle_roll"`**
         在 URDF 中，脚底碰撞体的节点都叫 `…_ankle_roll`，用来抓取脚部接触状态。

      - **`penalize_contacts_on = ["hip", "knee"]`**

        如果髋关节或膝关节部位发生不期望碰撞，将对对应环境打惩罚分数。

      - **`terminate_after_contacts_on = ["pelvis"]`**

         如果機器人的骨盆(`pelvis`)发生碰撞，则直接判定该环境 episode 终止。

      - **`self_collisions = 0` & `flip_visual_attachments = False`**

        - `self_collisions=0` 表示允许关节间自碰撞（默认是打开）；

        - `flip_visual_attachments=False`：沿用 URDF 原始的视觉模型坐标，不进行镜像翻转。

    + `rewards`

      + **`soft_dof_pos_limit = 0.9`**
         关节位置 “软” 极限阈值：如果关节超出 90%×(物理极限)，就开始给额外惩罚。

        **`base_height_target = 0.78`**
         期望基座高度：若机器人高度偏离 0.78 m，则按比例惩罚。

      + **嵌套 `scales`**（各项奖励/惩罚的权重）：

        - `tracking_lin_vel = 1.0`：跟随线速度 (x,y) 的正向奖励权重
        - `tracking_ang_vel = 0.5`：跟随角速度 (yaw) 的奖励
        - `lin_vel_z = -2.0`：惩罚在 z 方向上不期望的速度（鼓励保持水平）
        - `ang_vel_xy = -0.05`：惩罚绕 x/y 轴的角速度（鼓励机器人不左右翻滚）
        - `orientation = -1.0`：惩罚姿态偏离（鼓励机器人保持竖直）
        - `base_height = -10.0`：大幅度惩罚高度偏离 `0.78 m`
        - `dof_acc = -2.5e-7`：惩罚关节加速度（鼓励动作平滑）
        - `dof_vel = -1e-3`：惩罚关节速度过大
        - `feet_air_time = 0.0`：不关注脚离地时间奖励（置为 0）
        - `collision = 0.0`：将“碰撞”奖励项禁用（置 0），但脚下会由自定义的 `_reward_contact()` 等处理
        - `action_rate = -0.01`：惩罚动作更新率过快
        - `dof_pos_limits = -5.0`：惩罚关节位置超出“软限”
        - `alive = 0.15`：每步给一个“还活着”奖励，激励机器人不倒地
        - `hip_pos = -1.0`：惩罚髋关节位置偏离零—由自定义函数 `_reward_hip_pos()` 计算
        - `contact_no_vel = -0.2`：惩罚“接触时脚没有速度”（参见 `_reward_contact_no_vel()`）
        - `feet_swing_height = -20.0`：惩罚脚过低或过高（参见 `_reward_feet_swing_height()`）
        - `contact = 0.18`：奖励“臀部与命令相符的脚接触情况”（由 `_reward_contact()` 计算）

        以上权重在训练开始时会乘以仿真步长 `dt` 自动缩放，见 `LeggedRobot._prepare_reward_function()` 的实现。

    + `terrain`

      + **`mesh_type = "trimesh"`**
         指定使用 `terrain.py` 生成三角网格地形。`LeggedRobot.create_sim()` 里会调用 `convert_heightfield_to_trimesh(...)`，并在物理引擎中加载成 TriangleMesh。

    + `commands`

      + **`curriculum = True`**
         启用命令课表（Curriculum）：在 episode 初期只给机器人很简单的线速度范围，随着训练进度逐步扩大。继承自基类的默认命令维度仍为 4（`lin_vel_x, lin_vel_y, ang_vel_yaw, heading`）

+ `G1RoughCfgPPO`

  `LeggedRobotCfgPPO`的子类，用于配置PPO算法超参数

  + `policy`
    + `init_noise_std` 初始化时在 Actor 输出上打的高斯噪声
    + `actor_hidden_dims`： Actor 隐藏层维度
    + `critic_hidden_dims` ： Critic 隐藏层维度
    + `activation`: 激活函数
    + RNN 参数:
      + `rnn_hidden_size ` : RNN隐藏层维度
      + `rnn_num_layers ` ： RNN层数
  + `algorithm`
    + `entropy_coef` PPO 中的熵正则化系数
  + `runner`
    + `policy_class_name` 指定 Actor Critic 
    + `max_iterations` 指定策略更新次数
    + `experiment_name` 定义本次训练的日志/输出文件夹名前缀为 `g1`



`g1_env.py`

`LeggedRobot`的子类，用于定义环境逻辑

+ `_get_noise_scale_vec(cfg)` 

  将 `obs_buf` 中不同区段分别赋予不同的噪声强度，返还一个噪声向量

+ `_init_foot(self)`

  `feet_indices` 在 `LeggedRobot._create_envs()` 中通过 `asset.get_asset_rigid_body_names(...)` 把脚部在刚体列表中的索引读到 `self.feet_indices`。

  `_init_foot()` 会把所有并行环境的 `rigid_body_states`（形状 `(num_envs, num_bodies, 13)`）提取出来：

  - `feet_pos[:, i, :]`：第 i 条足端在世界坐标系下的 `(x, y, z)`
  - `feet_vel[:, i, :]`：第 i 条足端的线速度 `(vx, vy, vz)`

  这些信息后面会在计算特定奖励时使用，比如“脚在非支撑相位时高度要高于某阈值”、或“接触时脚速度太小要惩罚”等。

+ `_post_physics_step_callback(self)`

  + `update_feet_state(self)` 刷新脚部状态

  + `offset`  步态相位差

  + `phase` 步态相位

    每步 `self.episode_length_buf * self.dt` 表示当前 episode 已经过去的时间，然后对 `period` 取模，再除以 `period` 得到一个 [0,1) 区间内的相位。

    + `phase_left` 左腿相位
    + `phase_right` 右腿相位

    > 在观测拼接里，会把 `sin(2π·phase)` 和 `cos(2π·phase)` 作为额外输入，让策略知道“当前腿在哪个摆动或支撑相位”，形成一种隐式的“开环步态”信号。
    >
    > 在自定义奖励里，脚部实际接触状态 (contact) 要与“相位”对齐：如果此时相位 < 0.55 则表示“该腿应该处于支撑相位（stance）”，否则表示“摆动相位（swing）”。后续 `_reward_contact()` 就会把“脚在支撑相位没有接触”或“脚在摆动相位却接触地面”都算惩罚。

+ `compute_observations(self)`

  计算观测量

+ 自定义奖励函数

- `_reward_contact(self)`

  - 判断每条腿当前是否处于支撑（`leg_phase < 0.55`）或摆动（`leg_phase ≥ 0.55`）

    如果“处于支撑却没有接触”→(`contact=0`, `is_stance=1`) → `XOR=1` → `~XOR=0`（惩罚）

    如果“处于摆动却发生了接触”→(`contact=1`, `is_stance=0`) → `XOR=1` → `~XOR=0`（惩罚）

    只有“支撑并接触” or “摆动不接触”时给予 +1。

  - 最后 `res` 是对 `num_envs × feet_num` 的逐腿累加：一个环境每条腿都对齐的话就能得到 `feet_num` 的最大值，再乘以 `reward_scales['contact'] = 0.18`

- `_reward_feet_swing_height()`
  - `contact` 是一个 bool Tensor，指示哪些腿当前正与地面接触（法向力 > 1 N）。
  - 只在脚处于“摆动相位”时（即不接触时，~contact=1），才计算 `feet_pos[:,:,2] - 0.08` 的平方。
  - 目标是让脚在摆动段保持离地高度 ≈ 0.08 m，若过高或过低都给惩罚，最后按腿数求和、再乘以 `reward_scales['feet_swing_height'] = -20.0`

+ **`_reward_alive()`**

  只要 episode 没终止，就给每步 +1。结合 `reward_scales['alive']=0.15`，即每步固定奖励 +0.15（鼓励机器人不要跌倒）。

+ **`_reward_contact_no_vel()`**
  - 当脚与地面接触时（`contact=True`），`contact.unsqueeze(-1)` 扩维后就只保留相应腿的速度，否则乘以 0。
  - 取 `feet_vel` × `contact`（只有在接触时才非零），再做平方并按腿与维度求和，得到每个环境一个 scalar
  - 乘以 `reward_scales['contact_no_vel'] = -0.2`，惩罚“脚着地时速度过大”或“零速度时的异常接触”。
+ **`_reward_hip_pos()`**
  - 直接取腿部 DoF 列表 `[1,2,7,8]`（对应左右臀部俯仰、滚转等），令这些关节位置平方并按相应维度求和。
  - 乘以 `reward_scales['hip_pos'] = -1.0`，惩罚臀部关节偏离“零角度”，鼓励机器人保持躯干水平。







### train script

```python
python legged_gym/scripts/train.py --train=go1 --num_envs=2
```



### play script

reference： https://www.cnblogs.com/myleaf/p/18791990

```
python legged_gym/scripts/play.py --task=go1 --num_envs=2
```



### sim2sim script







## G1  with Isaac Lab Flow
