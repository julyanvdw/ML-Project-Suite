from FourRooms import FourRooms
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import sys

# HELPER FUNCTIONS
def generateStateSpace():
    # There are 11 x 11 = 121 possible states for the agent
    states = {}
    count = 0

    for x in range(1, 12):
        for y in range(1, 12):
            states[(x, y)] = count
            count += 1

    return states

def actionMapping(action):
    if action == 0:
        return FourRooms.UP
    elif action == 1:
        return FourRooms.DOWN
    elif action == 2:
        return FourRooms.LEFT
    elif action == 3:
        return FourRooms.RIGHT

# TEST FUNCTIONS
def plotMetrics(epochs, rewards_per_epoch):
    ave_reward = sum(rewards_per_epoch) / epochs

    # Plotting average reward
    plt.plot(rewards_per_epoch)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Average Steps per Episode: {:.2f}'.format(ave_reward))
    plt.savefig('plot.png')

# DECAY FUNCTIONS
def maxStepsDecay(initial_value, min_value, epoch, decay_rate):
    value = initial_value * math.pow(decay_rate, epoch)
    return max(value, min_value)

# REWARD FUNCTIONS
def rewardFunc(cellType):
    # If one of the packages are found - reward positively, but only ONCE (prevents cycles)
    rewarded_cells = [1, 2, 3]

    if cellType in rewarded_cells:
        rewarded_cells.remove(cellType)
        return 1000
    else:
        return -0.1

# TRAIN FUNCTION
def train(alpha, gamma, epsilon, max_step_count, min_step_value, max_step_decay_rate, epochs, stochastic_flag):
    # Initialising the room and creating state space
    fourRoomsObj = FourRooms('multi', stochastic=stochastic_flag)
    states = generateStateSpace()
    steps_per_epoch = []

    # Creating Q-table
    q_table = np.random.rand(3, 121, 4) #Initialises with random numbers for 121 states, 4 action per states, accross 3 spans (A span being the trip between 2 locations ie: agent -> 1, 1 -> 2, 2 -> 3)

    # FOR every EPOCH
    for epoch in range(epochs):
        # Clear the environment
        fourRoomsObj.newEpoch()
        step_count = 0
        
        done = False

        while not done:
            # Action Selection based on epsilon-greedy implementation
            current_state_number = states[fourRoomsObj.getPosition()]
            index_of_max_action = np.argmax(q_table[fourRoomsObj.getPackagesRemaining() - 1][current_state_number])

            if random.uniform(0, 1) < epsilon:
                # select a random action
                action = actionMapping(random.choice([0, 1, 2, 3]))
            else:
                # select based on q-table
                action = actionMapping(index_of_max_action)

            # Take action 
            cellType, newPos, packagesLeft, isTerminal = fourRoomsObj.takeAction(action)
            step_count += 1

            # Getting reward
            reward = rewardFunc(cellType)

            # Breaking out the loop once terminal state is reached OR if the run has continued for too long
            if isTerminal:
                done = True

            if (step_count >= maxStepsDecay(max_step_count, min_step_value, epoch, max_step_decay_rate)): 
                break
        
            # Calculating and updating the q-table
            old_value = q_table[fourRoomsObj.getPackagesRemaining() - 1][current_state_number][index_of_max_action]
            new_state_number =states[newPos]
            next_max = np.max(q_table[fourRoomsObj.getPackagesRemaining() - 1][new_state_number])

            new_value = (1-alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[fourRoomsObj.getPackagesRemaining() - 1][current_state_number][index_of_max_action] = new_value
        
        steps_per_epoch.append(step_count)

    # plotMetrics(epochs, steps_per_epoch)
    fourRoomsObj.showPath(index=-1, savefig="./image.png")


def main():
    # Check for stochastic
    s = False
    if len(sys.argv) == 2:
        if sys.argv[-1] == "-stochastic":
            s = True
        else: 
            print("Usage: python3 Scenario1.py -stochastic")
            return

    train(alpha=0.9, gamma=0.6, epsilon=0.001, max_step_count=1000, min_step_value=300, max_step_decay_rate= 0.95, epochs=100, stochastic_flag = s)

if __name__ == "__main__":
    main()

