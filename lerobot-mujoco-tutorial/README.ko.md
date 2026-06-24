# MuJoCo와 함께하는 LeRobot 튜토리얼
이 저장소는 커스텀 데이터셋에서 시연 데이터를 수집하고 비전-언어-행동 모델을 학습하거나 파인튜닝하기 위한 최소 예제를 담고 있습니다.

## 목차
- [:pencil: 설치](#설치)
- [:mega: 업데이트 및 계획](#업데이트--계획)
- [:video_game: 1. 시연 데이터 수집](#1-시연-데이터-수집)
- [:movie_camera: 2. 데이터 재생](#2-데이터-재생)
- [:fire: 3. Action-Chunking-Transformer(ACT) 학습](#3-action-chunking-transformeract-학습)
- [:pushpin: 4. ACT 배포](#4-정책-배포)
- [:floppy_disk: 5-6. 언어 조건 환경](#5-6-언어-조건-환경에서-데이터-수집과-시각화)
- [모델과 데이터셋](#모델과-데이터셋)
- [:zap: 7. pi_0 학습 및 배포](#7-pi_0-학습-및-배포)
- [:bulb: 8. SmolVLA 학습 및 배포](#8-smolvla-학습-및-배포)
- [:pencil: 감사의 글](#감사의-글)

## 설치
환경은 Python 3.10에서 테스트했습니다.

`pip install lerobot`으로 lerobot 패키지를 설치하는 것은 **권장하지 않습니다**. 오류가 발생할 수 있습니다.

mujoco 패키지 의존성과 lerobot을 설치합니다.
```
pip install -r requirements.txt
```
mujoco 버전이 **3.1.6**인지 확인하세요.

에셋 압축을 풉니다.
```
cd asset/objaverse
unzip plate_11.zip
```

### 업데이트 및 계획

:white_check_mark: 뷰어 업데이트

:white_check_mark: 다양한 언어 지시에 맞는 여러 머그컵과 접시 추가

:white_check_mark: pi_0 학습 및 추론 추가

:white_check_mark: SmolVLA 추가

## 1. 시연 데이터 수집

[1.collect_data.ipynb](1.collect_data.ipynb)를 실행하세요.

주어진 환경에서 시연 데이터를 수집합니다.
작업은 머그컵을 집어 접시 위에 놓는 것입니다. 머그컵이 접시 위에 있고, 그리퍼가 열려 있으며, 엔드 이펙터가 머그컵 위에 위치하면 환경은 성공으로 인식합니다.

<img src="./media/teleop.gif" width="480" height="360">

xy 평면 이동은 WASD, z축 이동은 RF, 기울이기는 QE, 나머지 회전은 방향키를 사용합니다.

SPACEBAR는 그리퍼 상태를 바꾸고, Z 키는 현재 에피소드 데이터를 버린 뒤 환경을 초기화합니다.

오버레이된 이미지는 다음과 같습니다.
- 오른쪽 위: 에이전트 뷰
- 오른쪽 아래: 에고센트릭 뷰
- 왼쪽 위: 왼쪽 측면 뷰
- 왼쪽 아래: 상단 뷰

데이터셋은 다음과 같이 구성됩니다.
```
fps = 20,
features={
    "observation.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.wrist_image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["state"], # x, y, z, roll, pitch, yaw
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["action"], # 6 joint angles and 1 gripper
    },
    "obj_init": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["obj_init"], # just the initial position of the object. Not used in training.
    },
},

```

이 과정은 `./demo_data` 폴더에 데이터셋을 생성하며, 구조는 다음과 같습니다.
```
.
├── data
│   └── chunk-000
│       ├── episode_000000.parquet
│       └── ...
├── meta
│   ├── episodes.jsonl
│   ├── info.json
│   ├── stats.json
│   └── tasks.jsonl
└──
```

편의를 위해 저장소에 [예제 데이터](./demo_data_example/)를 추가해 두었습니다.

## 2. 데이터 재생

[2.visualize_data.ipynb](2.visualize_data.ipynb)를 실행하세요.

<img src="./media/data.gif" width="480" height="360"></img>

재구성된 시뮬레이션 장면을 기반으로 액션을 시각화합니다.

메인 시뮬레이션은 액션을 재생합니다.

오른쪽 위와 오른쪽 아래에 오버레이된 이미지는 데이터셋에서 가져온 것입니다.

## 3. Action-Chunking-Transformer(ACT) 학습

[3.train.ipynb](3.train.ipynb)를 실행하세요.

**약 30~60분이 걸립니다**.

커스텀 데이터셋으로 ACT 모델을 학습합니다. 이 예제에서는 `chunk_size`를 10으로 설정합니다.

학습된 체크포인트는 `./ckpt/act_y` 폴더에 저장됩니다.

데이터셋에서 가져온 정답 액션과의 오차를 계산하여 정책을 평가할 수 있습니다.

<image src="./media/inference.png"  width="480" height="360">

<details>
    <summary>PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed</summary>
PicklingError가 발생하는 경우,

```
PicklingError: Can't pickle <function <lambda> at 0x131d1bd00>: attribute lookup <lambda> on __main__ failed
```

다음과 같이 `num_workers`를 0으로 설정하세요.

```
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0, # 4
    batch_size=64,
    shuffle=True,
    pin_memory=device.type != "cpu",
    drop_last=True,
)
```
</details>

## 4. 정책 배포

[4.deploy.ipynb](4.deploy.ipynb)를 실행하세요.

모델을 학습할 GPU가 없다면 [Google Drive](https://drive.google.com/drive/folders/1UqxqUgGPKU04DkpQqSWNgfYMhlvaiZsp?usp=sharing)에서 체크포인트를 다운로드할 수 있습니다.

<img src="./media/rollout.gif" width="480" height="360" controls></img>

학습된 정책을 시뮬레이션에서 배포합니다.


## 5-6. 언어 조건 환경에서 데이터 수집과 시각화

- [5.language_env.ipynb](5.language_env.ipynb): 키보드 원격 조작으로 데이터셋을 수집합니다. 명령은 첫 번째 환경과 동일합니다.
- [6.visualize_data.ipynb](6.visualize_data.ipynb): 수집된 데이터를 시각화합니다.


### 환경
**데이터**

<img src="./media/data_v2.gif" width="480" height="360" controls></img>


## 모델과 데이터셋
<table>
  <tr>
    <th> 모델 </th>
    <th> 데이터셋 </th>
    </tr>
    <tr>
        <td> <a href="https://huggingface.co/Jeongeun/omy_pnp_pi0"> 파인튜닝된 pi_0 </a></td>
        <td> <a href="https://huggingface.co/datasets/Jeongeun/omy_pnp_language"> dataset </a></td>
    </tr>
    <tr>
        <td> <a href="https://huggingface.co/Jeongeun/omy_pnp_smolvla"> 파인튜닝된 SmolVLA </td>
        <td> 같은 데이터셋</td>
    </tr>
</table>

## 7. pi_0 학습 및 배포
- [train_model.py](train_model.py): 학습 스크립트
- [pi0_omy.yaml](pi0_omy.yaml): 학습 설정 파일
- [7.pi0.ipynb](7.pi0.ipynb): 정책 배포



### 학습 스크립트
```
python train_model.py --config_path pi0_omy.yaml
```



### 학습된 정책의 롤아웃

<img src="./media/rollout2.gif" width="480" height="360" controls></img>


### 학습 로그

<image src="./media/wandb.png"  width="480" height="360">

### 설정 파일
```
dataset:
  repo_id: omy_pnp_language # Repository ID
  root: ./demo_data_language # Your root for data file!
policy:
  type : pi0
  chunk_size: 5
  n_action_steps: 5
  
save_checkpoint: true
output_dir: ./ckpt/pi0_omy <- Save directory
batch_size: 16
job_name : pi0_omy
resume: false 
seed : 42
num_workers: 8
steps: 20_000
eval_freq: -1 # No evaluation
log_freq: 50
save_checkpoint: true
save_freq: 10_000
use_policy_training_preset: true
  
wandb:
  enable: true
  project: pi0_omy
  entity: <your_wandb_entity>
  disable_artifact: true
```

## 8. SmolVLA 학습 및 배포

- [train_model.py](train_model.py): 학습 스크립트
- [smolvla_omy.yaml](smolvla_omy.yaml): 학습 설정 파일
- [8.smolvla.ipynb](8.smolvla.ipynb): 정책 배포



### 학습 스크립트
```
python train_model.py --config_path smolvla_omy.yaml
```



### 학습된 정책의 롤아웃

<img src="./media/rollout3.gif" width="480" height="360" controls></img>


### 학습 로그

<image src="./media/wandb2.png"  width="480" height="360">

### 설정 파일
```
dataset:
  repo_id: omy_pnp_language # Repository ID
  root: ./demo_data_language # Your root for data file!
policy:
  type : smolvla
  chunk_size: 5
  n_action_steps: 5
  device: cuda
  
save_checkpoint: true
output_dir: ./ckpt/smolvla_omy # Save directory
batch_size: 16
job_name : smolvla_omy
resume: false 
seed : 42
num_workers: 8
steps: 20_000
eval_freq: -1 # No evaluation
log_freq: 50
save_checkpoint: true
save_freq: 10_000
use_policy_training_preset: true
  
wandb:
  enable: true
  project: smolvla_omy
  entity: <your_wandb_entity>
  disable_artifact: true
```


## 감사의 글
- robotis-omy 매니퓰레이터 에셋은 [robotis_mujoco_menagerie](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie/tree/main)에서 가져왔습니다.
- [MuJoco Parser Class](./mujoco_env/mujoco_parser.py)는 [yet-another-mujoco-tutorial](https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3)을 수정한 것입니다.
- [lerobot examples](https://github.com/huggingface/lerobot/tree/main/examples)의 원본 튜토리얼을 참고했습니다.
- 접시와 머그컵 에셋은 [Objaverse](https://objaverse.allenai.org/)에서 가져왔습니다.
