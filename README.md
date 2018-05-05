# drl_lab
Facilitates experiments on deep reinforcement learning. Using OpenAi'gym 


## Install
```
docker exec -uroot -it container_id /bin/bash
apt update
apt install -y libsm6 libxext6 libxrender-dev
```

```
git clone https://github.com/chavdim/drl_lab.git
cd drl_lab
git checkout -b redo origin/redo
pip install scikit-image scikit-learn opencv-python imageio pygame
git clone https://github.com/openai/gym.git && cd gym
pip install -e . && cd ..
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ./ple && cd ple
pip install -e . && cd ..
git clone https://github.com/lusob/gym-ple.git && cd gym-ple
pip install -e . && cd ..
cp ./games/breakout_pygame.py ./ple/ple/games/
# add 'Breakout_pygame' in the list at line 5 of gym_ple/gym_ple/__init__.py
```

```
pip uninstall tensorflow-gpu
pip install tensorflow-gpu==1.5
apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
    libcudnn7=7.0.4.31-1+cuda9.0 \
    libcudnn7-dev=7.0.4.31-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*
```
