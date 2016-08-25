import numpy as np
import matplotlib.pyplot as plt
import copy
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--n_episode', '-n', default=100, type=int)
parser.add_argument('--epsilon', '-e', default=0.1, type=float)
parser.add_argument('--eval_epsilon', '-ev', default=0.01, type=float)
parser.add_argument('--gamma', '-g', default=0.95, type=float)
parser.add_argument('--alpha', '-a', default=0.1, type=float)
parser.add_argument('--eval_times', '-t', default=10.0, type=float)
parser.add_argument('--big_maze', '-b', default=0, type=int)


args = parser.parse_args()

n_episode = args.n_episode
epsilon = args.epsilon
eval_epsilon = args.eval_epsilon
gamma = args.gamma
alpha = args.alpha
eval_times = args.eval_times
big_maze = args.big_maze

#minimam 68steps
if big_maze == 1: 
	structure = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
						  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
						  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
						  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
						  [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
						  [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
						  [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
						  [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
						  [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
						  [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
						  [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						  [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
						  [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
						  [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
						  [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
						  [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
						  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
						  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
						  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#minimam 24steps
else:
	structure = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    	               	  [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
         	              [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            	          [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
                	      [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    	  [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                     	  [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
                     	  [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
                     	  [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                      	  [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                      	  [0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                      	  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

reward_table = np.zeros((structure.shape[0], structure.shape[1]))
reward_table[structure.shape[0]-2][structure.shape[1]-2] = 1

class Agent():
	def initialize(self):
		return np.array([1,1])

	def epsilon_greedy(self, position, V_table, epsilon):
		candidate, v_next = self.check(position)
		#random
		if np.random.rand() < epsilon:
			index = np.random.randint(candidate.size)
		#greedy
		else:
			v_max = np.amax(v_next)
			index_max = [x for x in range(v_next.size) if v_next[x]==v_max]
			index = np.random.choice(index_max) 					
		action = candidate[index]
		return action	

	def check(self, position):
		available = []
		v_next = []
		if structure[position[0]+1][position[1]] == 1:
 			available.append(0) #check down
			v_next.append(reward_table[position[0]+1][position[1]] + gamma*V_table[position[0]+1][position[1]])
		if structure[position[0]-1][position[1]] == 1:
			available.append(1) #check up
			v_next.append(reward_table[position[0]-1][position[1]] + gamma*V_table[position[0]-1][position[1]])
		if structure[position[0]][position[1]-1] == 1:
			available.append(2) #check left
			v_next.append(reward_table[position[0]][position[1]-1] + gamma*V_table[position[0]][position[1]-1])
		if structure[position[0]][position[1]+1] == 1:
			available.append(3) #check right
			v_next.append(reward_table[position[0]][position[1]+1] + gamma*V_table[position[0]][position[1]+1])
		available = np.array(available)
		v_next = np.array(v_next)
		return available, v_next

	def move(self, action, position):
		if action == 0: position += np.array([1,0]) #down
		if action == 1: position += np.array([-1,0]) #up
		if action == 2: position += np.array([0,-1]) #left
		if action == 3: position += np.array([0,1]) #right
		return position


	def finish(self, position):
		if np.array_equal(position, np.array([structure.shape[0]-2, structure.shape[1]-2])):flag = 1
		else: flag = 0
		return flag

class Value_table():
	def	initialize(self):	
		return np.zeros((structure.shape[0], structure.shape[1]))

	def update(self, prev_position, position, reward):
		td = reward + gamma*V_table[position[0]][position[1]] - V_table[prev_position[0]][prev_position[1]]
		V_table[prev_position[0]][prev_position[1]] += alpha*td

agent = Agent()
V = Value_table()
V_table = V.initialize()
eval_log = []

def evaluation():
	total_step = 0
	for i in range(int(eval_times)):
		position = agent.initialize()
		while(True):
			action = agent.epsilon_greedy(position, V_table, eval_epsilon)
			position = agent.move(action, position)
			total_step += 1
			finish_flag = agent.finish(position)
			if finish_flag:
				break
	eval_log.append(total_step/eval_times)
	print("Episode {} finished after {} steps".format(i_episode+1, total_step/eval_times))		

#main
for i_episode in range(n_episode):
	evaluation()
	position = agent.initialize()
	while(True):
		action = agent.epsilon_greedy(position, V_table, epsilon)
		prev_position = copy.deepcopy(position)
		position = agent.move(action, position)
		reward = reward_table[position[0]][position[1]]
		V.update(prev_position, position, reward)
		finish_flag = agent.finish(position)
		if finish_flag:
			break


#output
x = np.arange(n_episode)
y = np.array(eval_log)
plt.plot(x, y)
if big_maze == 1:
	plt.savefig('td_big/curve(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
else:
	plt.savefig('td/curve(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()	

sns.heatmap(V_table, linewidths=0.5)
plt.text(1.5, structure.shape[0]-1.5, "S", ha = 'center', va = 'center')
plt.text(structure.shape[1]-1.5, 1.5 , "G", ha = 'center', va = 'center')	
if big_maze == 1:
	plt.savefig('td_big/V_table(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
else:	
	plt.savefig('td/V_table(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()

sns.heatmap(structure, linewidths=0.5, cmap='Blues_r', cbar=False)
plt.text(1.5, structure.shape[0]-1.5, "S", ha = 'center', va = 'center')
plt.text(structure.shape[1]-1.5, 1.5 , "G", ha = 'center', va = 'center')
if big_maze == 1:
	plt.savefig('td_big/maze.png'.format(alpha, gamma, epsilon, n_episode))
else:
	plt.savefig('td/maze.png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()
