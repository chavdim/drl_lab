# Deep Reinforcement Learning Laboratory
Facilitates experiments on deep reinforcement learning. Using OpenAi's gym.


## Install
### Common
```bash
cd /path/to/your/workspace
git clone https://github.com/walkingmask/drl_lab.git
cd drl_lab
# after install gym-ple (at jupyter terminal)
cp games/breakout_pygame.py ple/ple/games/
vim gym-ple/gym_ple/__init__.py
# add 'Breakout_pygame' in the list at line 5
```

### Local
```bash
# require pyenv and pyenv-virtualenv
pyenv install anaconda3-5.0.0
pyenv virtualenv anaconda3-5.0.0 drl_lab
pyenv local drl_lab
pip install -U pip
pip install -r requirements.txt
git clone https://github.com/openai/gym.git && cd gym && pip install -e . && cd ..
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ./ple && cd ple && pip install -e . && cd ..
git clone https://github.com/lusob/gym-ple.git && cd gym-ple && pip install -e . && cd ..
```

### Local Docker(Mac)
```bash
# at local terminal
bash docker/local/run
open http://localhost:58888
```

### Remote Docker (with GeForce GT 730)
```bash
# at remote terminal
bash docker/remote_low/run
# at local terminal
open http://{remote_host}:58888
```

### Remote Docker (with GeForce GTX 1060)
```bash
# at remote terminal
bash docker/remote/run
# at local terminal
open http://{remote_host}:58888
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
- gcamの実装
    - simに組み込み
        - テスト、RLで
        - learning faseとtesting? faseがある
            - learning faseでは、modelのパラメータ(あるいはmodel)自体をガンガン保存して行く
                - 現状、model自体が軽いので、とりあえずぽんぽん保存して行っていいと思う
            - testing faseでは、保存した全てのパラメータでforを回す
                - episode_modeで
                - 10 epsくらいとか
                - epsの画像を全て保存
                - 保存した画像をagentに食わせてforward_propしてgcamして保存
            - 現状はflagで切り替えている
            - if test_agent:
- pixelcopter, breakout収束させる
    - reward.npyや、画像を確認
- expt.py
    - modeを追加
        - learn
        - run
            - test_agentをtrueにして、1epsだけ、画像保存など
- 最適化
    - 高速化できるならする


## Issues
- grad-camについて
    - QCNNを使ってMNISTをgcamして見たが、望ましい結果は得られなかった
    - 小さい画像に対して弱い？
    - guidedについては全くダメだった
    - VGG16以外で使いにくい？
    - どのlayerを選ぶかと言うハイパーパラメータが煩わしい
        - 論文だと最終畳み込み層だけど
- skimage.transform.rescaleについて
    - 戻り値がrange=(0.0, 1.0),dtype=np.float64になって帰ってくる
    - これは想定外の挙動
    - preserve_range=Trueでrangeは変えないでおけるぽい
        - envに適用済み
    - resizeも存在する
- 可視化関数をモジュール式にしていくつもの可視化を試せるように
