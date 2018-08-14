"""Upper Confidence Bound Basic Module
"""

###############################################################################
# Importing the libraries
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

###############################################################################
# Importing the dataset
###############################################################################
dataset = pd.read_csv(
    '/home/baxtor95/ML_Course/Projects/Part 6/S32/Ads_CTR_Optimisation.csv')

###############################################################################
# Defining Variables
###############################################################################
N = 10000
d = 10
numbers_of_selections = [0]*d
sums_of_rewards = [0]*d
average_reward = 0
delta_i = 0
upper_bound = 0
ads_selected = []
max_upper_bound = 0
ad = 0
reward = 0
total_reward = 0

###############################################################################
# Implementing UCB
###############################################################################

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) /
                                numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

###############################################################################
# Visualising the results
###############################################################################
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# Quit script
quit()
