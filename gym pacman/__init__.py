from gym.envs.registration import register

register(
	id='Pacman-v0',
	entry_point='gym_pacman.envs:PacmanEnv',
)