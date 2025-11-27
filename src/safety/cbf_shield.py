import numpy as np


class CBFShield:
    """
    Control Barrier Function (CBF) Shield mechanism.

    This class implements a safety filter that modulates the action
    proposed by the RL agent to ensure the drone stays within the safe set.

    Theoretical Basis:
    h(x) >= 0  (Safe Set)
    dot(h, x) + gamma * h(x) >= 0 (CBF Condition)
    """

    def __init__(self, safe_distance: float = 0.2, gamma: float = 1.0):
        """
        Initialize the safety shield.

        Args:
            safe_distance (float): Minimum allowed distance to obstacles (meters).
            gamma (float): Controller gain for the barrier function (aggressiveness).
        """
        self.safe_distance = safe_distance
        self.gamma = gamma

    def is_safe(self, state: np.ndarray, obstacle_pos: np.ndarray) -> bool:
        """Checks if the current state is within the safe set."""
        # state[0:3] is typically position (x, y, z)
        drone_pos = state[0:3]
        distance = np.linalg.norm(drone_pos - obstacle_pos)
        return distance > self.safe_distance

    def modify_action(
        self, action: np.ndarray, state: np.ndarray, obstacle_pos: np.ndarray
    ) -> np.ndarray:
        """
        Filters the RL action using a simplified barrier logic.

        If the drone is too close to an obstacle and moving towards it,
        we override the action to push it away.

        Args:
            action (np.ndarray): The action proposed by the RL agent (e.g., velocity or thrust).
            state (np.ndarray): The drone's current state vector.
            obstacle_pos (np.ndarray): The position of the nearest obstacle.

        Returns:
            np.ndarray: The safe action.
        """
        drone_pos = state[0:3]
        # Vector from obstacle to drone
        direction_vector = drone_pos - obstacle_pos
        distance = np.linalg.norm(direction_vector)

        # 1. Check Safety Barrier (Simple Euclidean distance for now)
        if distance < self.safe_distance:
            # DANGER: We are violating the safety constraint
            # Simple fallback: Generate a repulsion vector away from the obstacle
            repulsion = (direction_vector / distance) * 1.5  # Magnitude 1.5

            # We blend the RL action with the repulsion (or replace it entirely)
            # Ideally, this would be a QP (Quadratic Programming) solver.
            # For this MVP, we do a hard override if critical.
            safe_action = action + repulsion

            # Clip action to valid range [-1, 1] (standard in RL)
            return np.clip(safe_action, -1.0, 1.0)

        # 2. If safe, pass the RL action through unchanged
        return action
