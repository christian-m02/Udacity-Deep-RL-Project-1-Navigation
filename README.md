[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


This project is part of Udacity's Nanodegree on Deep Reinforcement Learning (https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).


# Project 1: Navigation

Goal of the exercise is to train an agent via Reinforcement Learning to navigate a large, square world and collect yellow bananas
while avoiding blue ones.


![Trained Agent][image1]


The exercise uses the Unity Machine Learning Agents Toolkit (https://github.com/Unity-Technologies/ml-agents)
which is an open-source Unity plugin.
It enables games and simulations to be used as an environment for training intelligent agents. Agents can be trained using
reinforcement learning through a simple-to-use Python API.

I will train and assess the DQN, Double DQN and Dueling DQN Agents.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the
goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the
agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are
available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. 

The ipython notebook contains the training and evaluation of the DQN, Double DQN and Dueling DQN Agents for performing this
task. For a discussion of the findings and assessment of the agents see the Report.pdf.



# Prerequisites

Running this notebook requires Python 3.5 (or higher), the Banana environment (see below) and the following Python libraries:

- NumPy
- PyTorch
- Unity Machine Learning Agents Toolkit
- OpenAI Gym



# To get started -

Below are the instructions on how to get started on this project as given in the original repository.
The original repository can be found here: https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this repository, in the `Udacity-Deep-RL-Project-1-Navigation/` folder, and unzip (or decompress) the file.


