# at top of file
import mujoco
import numpy as np
from dataclasses import dataclass

class SoccerKickEnv:
    def __init__(self, xml_path, max_steps=3000, dt=0.002, seed=0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        # Discrete action set: a bit richer, but not huge
        self.angles = np.deg2rad(np.array([-8, -4, 0, 4, 8], dtype=np.float32))
        self.speeds = np.array([8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float32)

        self.actions = [(th, v) for th in self.angles for v in self.speeds]

        self.site_target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.ball_body   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.goal_body   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,  "goal")

        # judge success on the plane of the goal (x ≈ goal_x)
        self.goal_x = float(self.model.body_pos[self.goal_body][0])
        self.success_radius = 0.35  # start a little forgiving for learnability

    def _set_target(self, pos_xyz):
        self.model.site_pos[self.site_target] = np.array(pos_xyz, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _sample_target(self):
        x = self.goal_x + self.rng.uniform(-0.05, 0.05)
        y = self.rng.uniform(-0.9, 0.9)
        z = self.rng.uniform(0.7, 1.2)
        return np.array([x, y, z], dtype=np.float64)

    def _ball_pos(self):
        return self.data.qpos[0:3].copy()

    def _ball_vel(self):
        return self.data.qvel[0:3].copy()

    def reset(self, target_random=True):
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        if target_random:
            self._set_target(self._sample_target())
        self.t = 0
        return self.model.site_pos[self.site_target].astype(np.float32).copy()

    def step(self, action_idx):
        """
        One-shot kick: convert (angle_yaw, speed) into an initial ball velocity,
        then simulate until the ball reaches the goal plane or comes to rest.
        Success = ball crosses x=goal_x BETWEEN the posts/crossbar.
        """
        # ----- action → velocity -----
        nA = len(self.actions)
        idx = int(action_idx) % nA
        yaw, speed = self.actions[idx]
        yaw = float(yaw); speed = float(speed)

        # elevation lifted so the ball reaches the mouth height
        elev = np.deg2rad(20.0)

        # small variability to avoid trivial 100% success
        yaw  += np.deg2rad(np.random.uniform(-1.0, 1.0))   # ±1°
        speed *= np.random.uniform(0.95, 1.05)             # ±5%

        # components
        vx = float(speed * np.cos(elev) * np.cos(yaw))
        vy = float(speed * np.cos(elev) * np.sin(yaw))
        vz = float(speed * np.sin(elev))

        # reset and apply initial ball velocity "impulse"
        self.data.ctrl[:] = 0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.data.qvel[0:3] = np.array([vx, vy, vz], dtype=np.float64)

        # ----- simulate flight until plane crossing or stop -----
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

        # ----- goal bounds (between posts) -----
        goal_y_half = 1.0   # posts at ±1.0 m
        goal_z_min  = 0.20  # allow low shots
        goal_z_max  = 1.60  # crossbar-ish

        # ----- reward -----
        if crossed and cross_pos is not None:
            y, z = float(cross_pos[1]), float(cross_pos[2])
            success = (-goal_y_half <= y <= goal_y_half) and (goal_z_min <= z <= goal_z_max)

            if success:
                reward = 5.0
            else:
                # graded penalty for near misses (horizontal weighted more)
                z_mid = 0.5 * (goal_z_min + goal_z_max)
                dy = abs(y) / goal_y_half
                dz = abs(z - z_mid) / (0.5 * (goal_z_max - goal_z_min))
                miss_penalty = 0.6 * dy + 0.4 * dz
                reward = 0.8 - 3.0 * miss_penalty
        else:
            # didn't reach plane → penalize x shortfall
            end = self._ball_pos()
            dist_plane = abs(self.goal_x - float(end[0]))
            reward = -0.3 * dist_plane - 0.5
            success = False

        info = {"success": bool(success), "cross_pos": None if cross_pos is None else cross_pos.copy()}
        done = True
        return self._get_obs(), float(reward), done, info





    def reset_ball(self):
        """Reposition the ball back to its start position."""
        ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        # default ball start pos (adjust if needed)
        start_pos = np.array([-0.35, 0, 0.11])
        self.model.body_pos[ball_body_id] = start_pos
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[:3] = start_pos
        self.data.qvel[:3] = 0
        mujoco.mj_forward(self.model, self.data)

    def reset_kicker(self):
        """Return the kicker arm back to rest position."""
        hinge_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        hinge_qposadr = self.model.jnt_qposadr[hinge_id]
        self.data.qpos[hinge_qposadr] = 0.0
        self.data.qvel[hinge_qposadr] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def manual_control(self):
        """Interactive control loop with working keyboard input."""
        from mujoco import viewer
        print("🎮 Interactive Controls:")
        print("  [SPACE] = Kick")
        print("  [R] = Reset ball + kicker")
        print("  [T] / [G] = Increase / Decrease torque\n")

        torque = 0.8
        with viewer.launch(self.model, self.data) as v:  # <-- not passive
            while v.is_running():
                # Check keypresses
                key = v.user_key_press
                if key == ' ':  # Kick
                    for _ in range(60):
                        self.data.ctrl[0] = torque
                        mujoco.mj_step(self.model, self.data)
                        v.sync()
                    self.data.ctrl[0] = 0
                elif key in ('r', 'R'):
                    self.reset_ball()
                    self.reset_kicker()
                    print("Reset ball and kicker.")
                elif key in ('t', 'T'):
                    torque = min(torque + 0.1, 2.0)
                    print(f"Torque: {torque:.2f}")
                elif key in ('g', 'G'):
                    torque = max(torque - 0.1, 0.1)
                    print(f"Torque: {torque:.2f}")
                # Auto-reset when ball slows down near rest
                vel = np.linalg.norm(self.data.qvel[0:3])
                if vel < 0.05:
                    self.reset_ball()
                    self.reset_kicker()

                mujoco.mj_step(self.model, self.data)
                v.sync()

    def _get_obs(self):
    # Return relative vector from ball to goal center (helps learning direction)
        ball = self._ball_pos()
        goal_center = np.array([self.goal_x, 0.0, 1.0], dtype=np.float32)
        rel = goal_center - ball
        return rel.astype(np.float32)



    def render(self):
        from mujoco import viewer
        viewer.launch(self.model, self.data)

