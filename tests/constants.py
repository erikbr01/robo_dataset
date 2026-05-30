"""Shared test constants for synthetic dataset generation."""

N_EPISODES = 5
EP_LENGTHS = [10, 8, 12, 7, 15]
H, W, C = 32, 32, 3
ACTION_DIM = 7
OBS_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
LOWDIM_KEY = "robot0_eef_pos"
LOWDIM_DIM = 3
