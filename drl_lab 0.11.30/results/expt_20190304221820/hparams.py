env_hparams = {'action': {'excluded_actions': []},
 'env_id': 'Breakout_pygame-v0',
 'observation': {'normalize': False, 'opt_flow': False, 'rescaled_shape': []}}
run_hparams = {'max_steps': 1000, 'num_runs': 1, 'save_at': None, 'verbose': True}
nn_hparams = {'layers': [['conv', 30, 8, 4],
            ['conv', 40, 4, 3],
            ['conv', 60, 3, 1],
            ['gap'],
            ['fc', 512]],
 'learn_rate': 5e-05,
 'optimizer': 'RMSprop',
 'saved_model': None}
agent_hparams = {'batch_size': 32,
 'final_epsilon': 0.1,
 'initial_epsilon': 1.0,
 'reward_decay': 0.99,
 'target_q_network_update_freq': 10}
