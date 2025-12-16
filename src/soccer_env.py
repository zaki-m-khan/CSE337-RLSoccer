import mujoco
import numpy as np

class SoccerKickEnv:
    def __init__(self, xml_path, max_steps=3000, dt=0.002, seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        # Discrete action set: 5 angles x 5 speeds = 25 actions
        self.angles = np.deg2rad(np.array([-8, -4, 0, 4, 8], dtype=np.float32))
        self.speeds = np.array([8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float32)
        self.actions = [(th, v) for th in self.angles for v in self.speeds]

        self.site_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.ball_body   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.goal_body   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.goal_x = float(self.model.body_pos[self.goal_body][0])

    def _ball_pos(self):
        return self.data.qpos[0:3].copy()

    def _ball_vel(self):
        return self.data.qvel[0:3].copy()

    def reset(self, target_random=True, seed=None):
        """Reset environment. Optionally reseed RNG for reproducibility."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if target_random:
            x = self.goal_x + self.rng.uniform(-0.05, 0.05)
            y = self.rng.uniform(-0.9, 0.9)
            z = self.rng.uniform(0.7, 1.2)
            self.model.site_pos[self.site_target] = np.array([x, y, z], dtype=np.float64)
            mujoco.mj_forward(self.model, self.data)
        return self.model.site_pos[self.site_target].astype(np.float32).copy()

    def step(self, action_idx):
        """
        One-shot kick: convert (angle_yaw, speed) into an initial ball velocity,
        then simulate until the ball reaches the goal plane or comes to rest.
        Success = ball crosses x=goal_x BETWEEN the posts/crossbar.
        """
        nA = len(self.actions)
        idx = int(action_idx) % nA
        yaw, speed = self.actions[idx]
        yaw = float(yaw); speed = float(speed)
        yaw_deg = np.rad2deg(yaw)  # Store original yaw for logging

        elev = np.deg2rad(20.0)
        
        # Use seeded RNG for reproducibility (not np.random.uniform!)
        yaw  += np.deg2rad(self.rng.uniform(-1.0, 1.0))
        speed *= self.rng.uniform(0.95, 1.05)

        vx = float(speed * np.cos(elev) * np.cos(yaw))
        vy = float(speed * np.cos(elev) * np.sin(yaw))
        vz = float(speed * np.sin(elev))

        self.data.ctrl[:] = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[0:3] = np.array([vx, vy, vz], dtype=np.float64)

        crossed = False
        cross_pos = None
        for _ in range(self.max_steps):
            mujoco.mj_step(self.model, self.data)
            p = self._ball_pos()
            v = self._ball_vel()
            if (p[0] >= self.goal_x) and not crossed:
                crossed = True
                cross_pos = p.copy()
                break
            if np.linalg.norm(v) < 0.05 and p[2] < 0.12:
                break

        # Goal bounds
        goal_y_half = 1.0   # posts at ±1.0 m
        goal_z_min  = 0.20  # allow low shots
        goal_z_max  = 1.60  # crossbar-ish

        # Compute reward and outcome
        if crossed and cross_pos is not None:
            y, z = float(cross_pos[1]), float(cross_pos[2])
            success = (-goal_y_half <= y <= goal_y_half) and (goal_z_min <= z <= goal_z_max)
            if success:
                reward = 5.0
                outcome = "goal"
            else:
                z_mid = 0.5 * (goal_z_min + goal_z_max)
                dy = abs(y) / goal_y_half
                dz = abs(z - z_mid) / (0.5 * (goal_z_max - goal_z_min))
                miss_penalty = 0.6 * dy + 0.4 * dz
                reward = 0.8 - 3.0 * miss_penalty
                outcome = "miss"
        else:
            end = self._ball_pos()
            dist_plane = abs(self.goal_x - float(end[0]))
            reward = -0.3 * dist_plane - 0.5
            success = False
            outcome = "miss"

        # Extract goal crossing coordinates for logging
        y_goal = float(cross_pos[1]) if cross_pos is not None else None
        z_goal = float(cross_pos[2]) if cross_pos is not None else None

        info = {
            "success": bool(success),
            "outcome": outcome,
            "cross_pos": cross_pos,
            "y_goal": y_goal,
            "z_goal": z_goal,
            "yaw_deg": yaw_deg,
            "speed": float(self.actions[idx][1])
        }
        return self._get_obs(), float(reward), True, info

    def _get_obs(self):
        """Return relative vector from ball to goal center (helps learning direction)."""
        ball = self._ball_pos()
        goal_center = np.array([self.goal_x, 0.0, 1.0], dtype=np.float32)
        return (goal_center - ball).astype(np.float32)

    def render(self):
        """Open MuJoCo viewer (only works locally, not in Colab)."""
        from mujoco import viewer
        viewer.launch(self.model, self.data)
