# go2_env.py
# Custom GO2 environment with trot-phase signals for diagonal pairs and previous observation context

from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch
import torch
import numpy as np

class GO2Robot(LeggedRobot):
    """
    GO2 environment subclass that provides:
      - diagonal trot-phase signals (for FL+RR and FR+RL pairs)
      - inclusion of previous observation for temporal context
      - a diagonal contact reward to encourage trot
    """

    def _init_foot(self):
        # Wrap rigid body state tensor and initialize foot kinematics
        tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(tensor).view(self.num_envs, -1, 13)
        self.update_feet_state()

    def update_feet_state(self):
        # Refresh and extract foot positions and velocities
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        states = self.rigid_body_states[:, self.feet_indices, :]
        self.feet_pos = states[:, :, :3]
        self.feet_vel = states[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()
        # Buffer to hold previous raw observation (48-dim)
        obs_dim = self.cfg.env.num_observations
        self.prev_obs = torch.zeros(self.num_envs, obs_dim, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        # Extend base noise vector by 4 phase dims and previous obs dims
        base_vec = super()._get_noise_scale_vec(cfg)
        extra = torch.zeros(4 + self.prev_obs.shape[-1], device=self.device)
        return torch.cat([base_vec, extra], dim=0)

    def _post_physics_step_callback(self):
        # Update foot kinematics
        self.update_feet_state()
        # Compute trot phases for diagonal pairs
        period = 0.8
        offset = 0.5
        t = ((self.episode_length_buf * self.dt) % period) / period
        self.phase_primary = t
        self.phase_secondary = (t + offset) % 1
        return super()._post_physics_step_callback()

    def compute_observations(self):
        # Compute default observation (48-dim)
        super().compute_observations()
        # Save raw obs for next step
        raw_obs = self.obs_buf.clone()
        # Phase signals (sin, cos) for two phases
        sin_p = torch.sin(2 * np.pi * self.phase_primary).unsqueeze(1)
        cos_p = torch.cos(2 * np.pi * self.phase_primary).unsqueeze(1)
        sin_s = torch.sin(2 * np.pi * self.phase_secondary).unsqueeze(1)
        cos_s = torch.cos(2 * np.pi * self.phase_secondary).unsqueeze(1)
        phase_vec = torch.cat([sin_p, cos_p, sin_s, cos_s], dim=-1)
        # Concatenate: raw_obs (48), phase_vec (4), prev_obs (48)
        self.obs_buf = torch.cat([raw_obs, phase_vec, self.prev_obs], dim=-1)
        # Update prev_obs
        self.prev_obs = raw_obs
        # No privileged observations
        self.privileged_obs_buf = None
        # Add noise if enabled
        if self.cfg.noise.add_noise:
            if not hasattr(self, 'noise_scale_vec'):
                self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_contact_trot(self):
        # Reward correct diagonal foot contacts aligned with phases
        contacts = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0)
        res = torch.zeros(self.num_envs, device=self.device)
        primary_mask = self.phase_primary < 0.5
        secondary_mask = self.phase_secondary < 0.5
        # feet order: [FL, FR, RL, RR]
        # primary diagonal: FL(0) + RR(3)
        res += (contacts[:, 0] & primary_mask).float()
        res += (contacts[:, 3] & primary_mask).float()
        # secondary diagonal: FR(1) + RL(2)
        res += (contacts[:, 1] & secondary_mask).float()
        res += (contacts[:, 2] & secondary_mask).float()
        return res

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # contact = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > force_thresh)
        contact = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1)
        contact_vel = self.feet_vel * contact.unsqueeze(-1)
        return torch.sum(contact_vel.square(), dim=(1,2))

    def _reward_feet_swing_height(self):
        # contact = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > force_thresh)
        contact = (torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1 )
        # swing_error = (self.feet_pos[:,:,2] - target_height).square() * (~contact)
        swing_error = (self.feet_pos[:,:,2] - 0.02).square() * (~contact)
        return swing_error.sum(dim=1)

    # def _reward_hip_pos(self):
    #     hip_regs = self.dof_pos[:, []]
    #     return torch.sum(hip_regs.square(), dim=1)
