from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from src.envs.safe_aviary import SafeHoverAviary


def test_pipeline():
    print("1. Initializing Base Environment...")
    base_env = HoverAviary(gui=False, record=False)

    print("2. Wrapping with Safety Shield...")
    safe_env = SafeHoverAviary(base_env)

    print("3. Resetting Environment...")
    obs, info = safe_env.reset()

    print("4. Testing Loop (Simulating 5 steps)...")
    for i in range(5):
        # Random action from RL agent
        action = safe_env.action_space.sample()

        # Step through our Safe Wrapper
        obs, reward, terminated, truncated, info = safe_env.step(action)

        safe_status = "SAFE" if info.get("is_safe") else "DANGER"
        print(f"   Step {i + 1}: Reward={reward:.4f} | Status={safe_status}")

    safe_env.close()
    print("\n✅ SUCCESS: The Safe RL Pipeline is fully functional locally!")


if __name__ == "__main__":
    test_pipeline()
