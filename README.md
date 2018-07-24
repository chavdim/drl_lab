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
- 用語の定義
    - iterations?
    - interval?
- expt.py
    - modeを追加
        - learn
        - run
            - test_agentをtrueにして、1epsだけ、画像保存など
- 役割の切り分け
    - models.py: CNN model
    - env.py   : Environment wrappers (state preprocessing)
    - agents.py: RL Agents
    - memory.py: memory for experience replay
    - sim.py   : env+agent+memory
    - expt.py  : sim+run_hparams+save
    - gcam.py  : Grad-CAM
    - main     : args
- 最適化
    - 高速化できるならする
- Action周り
    - せっかくAction作ってるのだから
    - action_indexisとか使わないでそれ使えないか？
- epsilonの調整
    - 28000stepで0.04まで下がってしまってる
    - もう少し下げ方調整していい感じにできないだろうか


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
- Q値は0-1じゃないよね
    - linear出力
        - パッと見た限り -0.xx ~ 1.xx あった
    - vgg16とかは0-1(softmax)
    - それによる影響は？ありそう
        - あった
        - gcamにかける前にmodelのlayers[-1].activationをsoftmaxに変更
        - gcamの結果めっちゃ変わった
        - けど、softmaxに変えていいものだろうか？
- grad-camはnegative contributionも反映してる？
    - つまり他クラスに分類される根拠を青くハイライトする？
    - 論文にはそう書いてあるぽい
    - けど見た感じそうなってるか微妙
    - Q値だからなのか？

