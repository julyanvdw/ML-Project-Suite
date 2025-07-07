# Julyan van der Westhuizen - VWSJUL003
# RL Assignment - Scenario 1 - Single Package Pickup

from FourRooms import FourRooms
import numpy as np
import random
import math
import sys

def generateStateSpace():
    # There are 11 x 11 = 121 possible states
    # This means that in the q-table - there are 4 actions associated to each of the states (states labelled from 0 .. 120)
    # To generate the state space and to associate a state with an x,y position - we define a dictionary where the x,y position tuple is the key to some value
    # once we have a particular x,y position tuple - we can retrieve the state number and hence index into the q-table

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

def train(alpha, gamma, epsilon, decay_rate, epochs, stochastic_flag):
    generateStateSpace()
    # Initialising the room and creating state space
    fourRoomsObj = FourRooms('simple', stochastic=stochastic_flag)
    states = generateStateSpace()

    initial_pos = fourRoomsObj.getPosition()

    # Creating Q-table
    q_table = np.zeros([121, 4]) # 11 x 11 = 121 possible states (positions) and 4 possible actions

    # Penalty Intialisation
    penalties, steps = 0, 0

    # FOR every EPOCH
    for epoch in range(epochs):
        # Clear the environment
        fourRoomsObj.newEpoch()
        run_epsilon = epsilon
        run_gamma = gamma
        run_alpha = alpha

        # Trying to find the box once
        done = False

        while not done:
            # Action Selection based on epsilon-greedy implementation
            current_state_number = states[fourRoomsObj.getPosition()]
            index_of_max_action = np.argmax(q_table[current_state_number])

            if random.uniform(0, 1) < run_epsilon:
                # select a random action
                action = actionMapping(random.choice([0, 1, 2, 3]))
            else:
                # select based on q-table
                action = actionMapping(index_of_max_action)

            # Take action 
            cellType, newPos, packagesLeft, isTerminal = fourRoomsObj.takeAction(action)

            # Breaking out the loop once terminal state is reached
            if isTerminal:
                done = True

            # Define a reward structure
            if isTerminal:
                reward = 10
            elif cellType == 0:
                reward = -0.1

            # Calculating and updating the q-table
            old_value = q_table[current_state_number][index_of_max_action]
            new_state_number =states[newPos]
            next_max = np.max(q_table[new_state_number])

            new_value = old_value + run_alpha * (reward + run_gamma*next_max - old_value)
            q_table[current_state_number][index_of_max_action] = new_value

        # Decay of param
        run_epsilon = run_epsilon * math.exp(-decay_rate * epoch)
        run_gamma = run_gamma * math.exp(-decay_rate * epoch)
        run_alpha = run_alpha * math.exp(-decay_rate * epoch)

    fourRoomsObj.showPath(index=-1, savefig="./image.png")

def main():
    # Check for Stochastic
    s = False
    if len(sys.argv) == 2:
        if sys.argv[-1] == "-stochastic":
            s = True
        else: 
            print("Usage: python3 Scenario1.py -stochastic")
            return

    train(alpha=0.8, gamma=0.4, epsilon=0.01, decay_rate=0.01, epochs = 100, stochastic_flag=s)
   

    
if __name__ == "__main__":
    main()
