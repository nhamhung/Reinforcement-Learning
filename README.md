# Reinforcement-Learning

- This is a sample code for my work done at PSA in order to apply DRL to container re-shuffling problem. 

## Rationale

As PSA is a transshipment hub, there are many containers lying in her terminals waiting for the arriving vessels so that they can be unloaded onto the vessels. The issue is that due to various reasons, these containers may be currently stacked in non-optimal orders and when their vessels arrive, it costs additional waiting time for the containers to be re-shuffled.

As such, it would benefit greatly if these containers are pre-shuffled beforehand. 

Currently, this job is done by human operators one slot of containers at a time, so the aim for this project is to have a RL agent which can take into account operators' intuition and perform this job for them automatically and as optimally as possible.

<img width="1101" alt="image" src="https://user-images.githubusercontent.com/43290909/159135050-feb70318-17a5-4695-a25f-d8b375608099.png">


## Entry point

There are 2 versions of the same model being used using RLLib and TF-Agents:

1. `combined_model.py` is the model with Ray's RLLib which is more updated
2. `main.py` is the model with TF-Agents

## Frameworks used

1. [RLLib](https://docs.ray.io/en/latest/rllib.html)
2. [TF-Agents](https://www.tensorflow.org/agents)

