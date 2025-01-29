# Starting code for course project of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1yA9Ta4rWlh2YcRfhtbeJp0TpS-2nUsKK8-wpp3iEj0M/edit?usp=sharing)

## How to Use

Registered Gym environments: 
- **CustomHopper-source-v0**: hopper defined in the source domain.
- **CustomHopper-target-v0**: hopper defined in the target domain.
- **CustomHopper-udr-v0**: hopper supporting UDR during training.
- **CustomHopper-source-adr-v0**: hopper supporting ADR during training.
- **HalfCheetah-source-v3**: half cheetah defined in the source domain.
- **HalfCheetah-target-v3**: half cheetah defined in the target domain.
- **HalfCheetah-source-udr-v3**: half cheetah supporting ADR during training.
- **HalfCheetah-source-adr-v3**: half cheetah supporting ADR during training.

Files: 
- `task1.py`: just to test the different environment setup.
- `task2.py`: can be used to train the model in any source, target or UDR enviroment. `--env env_id` to specify the environment to use. `--algo` to specify which algorithm to use during training/testing (supported values `ppo` and `sac`). `--test model_path` to test any of the trained model on the desired environment.
- `task3.py`: automatically trains and tests the following combination on the Hopper environment source->source, source->target, target->target.
- `task4.py`: automatically trains  and tests UDR on the Hopper environment sourceUDR->source, sourceUDR->target.
- `train_hopper_adr.py` and `train_cheetah_adr.py`: automatically train a models on Hopper (PPO) and Cheetah (SAC) environment with ADR integration. `task2.py` can be used to test the trained model.
- `train_multiple_models`: train N models on the specified env and domain. `--domain domain_name` possible values source, target, udr, adr. `--env env_name` possible values hopper or cheetah.
- `train_multiple_models`: test N previosly trained models on the specified env and domain. `--model model_path` possible values source, target, udr, adr. `--env env_name` possible values hopper or cheetah. Not all models combination are already trained.
- `models/`: contains a collection of already trained models both on Hopper and Cheetah environment.
- `automatic_domain_randomization.py`: contains the class that define the logic of Automatic Domain Randomization (ADR).


## Getting started

Before starting to implement your own code, make sure to:
1. read and study the material provided (see Section 1 of the assignment)
2. read the documentation of the main packages you will be using ([mujoco-py](https://github.com/openai/mujoco-py), [Gym](https://github.com/openai/gym), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local (Linux)

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

**Dependencies**
- Install MuJoCo and the Python Mujoco interface following the instructions here: https://github.com/openai/mujoco-py
- Run `pip install -r requirements.txt` to further install `gym` and `stable-baselines3`.

Check your installation by launching `python test_random_policy.py`.


### 2. Local (Windows)
As the latest version of `mujoco-py` is not compatible for Windows explicitly, you may:
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- (not recommended) Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) of `mujoco-py`.
- (not recommended) Stick to the Google Colab template (see below), which runs on the browser regardless of the operating system. This option, however, will not allow you to render the environment in an interactive window for debugging purposes.


### 3. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo.
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.
