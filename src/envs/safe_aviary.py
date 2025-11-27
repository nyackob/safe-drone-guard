import gymnasium as gym
import numpy as np
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from src.safety.cbf_shield import CBFShield


class SafeHoverAviary(gym.Wrapper):
    """
    A Gym Wrapper that adds a Safety Layer (CBF Shield) to the standard Drone environment.
    """

    def __init__(self, env: HoverAviary):
        super().__init__(env)
        self.env = env
        # Initialize our custom Safety Shield
        self.shield = CBFShield(safe_distance=0.3, gamma=0.5)
        # Dummy obstacle position for testing (e.g., placed at origin or nearby)
        self.obstacle_pos = np.array([0.0, 0.0, 0.5])

    def step(self, action):
        """
        Overwrites the step method to filter actions.
        """
        # 1. Get current state (simplified, usually we get this from last obs)
        # In gym-pybullet-drones, state access can be complex,
        # here we cheat slightly by accessing internal pos for the demo
        drone_pos = self.env.pos[0]
        drone_state = np.hstack(
            [drone_pos, np.zeros(9)]
        )  # Simplified state construction

        # 2. Filter the action through the shield
        safe_action = self.shield.modify_action(
            action=action, state=drone_state, obstacle_pos=self.obstacle_pos
        )

        # 3. Execute the SAFE action in the real environment
        obs, reward, terminated, truncated, info = self.env.step(safe_action)

        # 4. Add safety info to the logs (Crucial for XAI)
        is_safe_now = self.shield.is_safe(drone_state, self.obstacle_pos)
        info["is_safe"] = is_safe_now

        # Optional: Penalize reward if shield had to intervene
        if not np.array_equal(action, safe_action):
            reward -= 0.5  # Penalty for unsafe proposal

        return obs, reward, terminated, truncated, info
