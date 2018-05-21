# drl_lab
Facilitates experiments on deep reinforcement learning. Using OpenAi'gym 


## Install
### Common
```bash
cd /path/to/your/workspace
git clone https://github.com/walkingmask/drl_lab.git
cd drl_lab
```

### Local
```bash
# require pyenv and pyenv-virtualenv
pyenv install anaconda3-5.0.0
pyenv virtualenv anaconda3-5.0.0 drl_lab
pyenv local drl_lab
pip install -U pip
conda update -y conda
pip install -r requirements.txt
git clone https://github.com/openai/gym.git && cd gym && pip install -e . && cd ..
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ./ple && cd ple && pip install -e . && cd ..
git clone https://github.com/lusob/gym-ple.git && cd gym-ple && pip install -e . && cd ..
cp games/breakout_pygame.py ple/ple/games/
vim gym-ple/gym-ple/__init__.py
# add 'Breakout_pygame' in the list at line 5
```

### Local Docker(Mac)
```bash
# at local terminal
bash docker/local/run
open http://localhost:58888
# at jupyter terminal
cp drl_lab/games/breakout_pygame.py ple/ple/games/
vim gym-ple/gym-ple/__init__.py
# add 'Breakout_pygame' in the list at line 5
```

### Remote Docker (with GeForce GT 730)
```bash
# at remote terminal
bash docker/remote_low/run
# at local terminal
open http://{remote_host}:58888
# at jupyter terminal
cp drl_lab/games/breakout_pygame.py ple/ple/games/
vim gym-ple/gym-ple/__init__.py
# add 'Breakout_pygame' in the list at line 5
```

### Remote Docker (with GeForce GTX 1060)
```bash
# at remote terminal
bash docker/remote/run
# at local terminal
open http://{remote_host}:58888
# at jupyter terminal
cp drl_lab/games/breakout_pygame.py ple/ple/games/
vim gym-ple/gym-ple/__init__.py
# add 'Breakout_pygame' in the list at line 5
```


## Tests
```bash
python -m unittest discover
```


## Run
```bash
python main.py --help                   # show helps
python main.py                          # run as default
python main.py --hparams ./hparams.py   # run using hparams.py
```


## TODO
- expt.py
    - modeを追加
        - learn
        - run
            - test_agentをtrueにして、1epsだけ、画像保存など
- 用語の定義
    - iterations?
    - interval?
- 役割の切り分け
    - models.py: CNN model
    - env.py   : Environment wrappers (state preprocessing)
    - agents.py: RL Agents
    - memory.py: memory for experience replay
    - sim.py   : env+agent+memory
    - expt.py  : sim+run_hparams+save
    - gcam.py  : Grad-CAM
    - main     : args
- pixelcopter, breakout収束させる
    - reward.npyや、画像を確認
- gcamの実装
    - テスト、画像で確認
    - simに組み込み
- 最適化
    - 高速化できるならする
- Action周り
    - せっかくAction作ってるのだから
    - action_indexisとか使わないでそれ使えないか？
