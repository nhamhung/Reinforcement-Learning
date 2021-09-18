# Reinforcement-Learning

- This is a sample code for my work done at PSA in order to apply DRL to container re-shuffling problem. 

## Rationale

As PSA is a transshipment hub, there are many containers lying in her terminals waiting for the arriving vessels so that they can be unloaded onto the vessels. The issue is that due to various reasons, these containers may be currently stacked in non-optimal orders and when their vessels arrive, it costs additional waiting time for the containers to be re-shuffled.

As such, it would benefit greatly if these containers are pre-shuffled beforehand. 

Currently, this job is done by human operators one slot of containers at a time, so the aim for this project is to have a RL agent which can take into account operators' intuition and perform this job for them automatically and optimally as much as possible
