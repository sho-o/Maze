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

num_of_action = 4

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

	def epsilon_greedy(self, position, Q_table, epsilon):
		candidate, q = self.check(position)
		#random
		if np.random.rand() < epsilon:
			index = np.random.randint(candidate.size)
		#greedy
		else:
			q_max = np.amax(q)
			index_max = [x for x in range(q.size) if q[x]==q_max]
			index = np.random.choice(index_max) 					
		action = candidate[index]
		return action	

	def check(self, position):
		available = []
		q = []
		if structure[position[0]+1][position[1]] == 1:
 			available.append(0) #check down
			q.append(Q_table[position[0]][position[1]][0])
		if structure[position[0]-1][position[1]] == 1:
			available.append(1) #check up
			q.append(Q_table[position[0]][position[1]][1])
		if structure[position[0]][position[1]-1] == 1:
			available.append(2) #check left
			q.append(Q_table[position[0]][position[1]][2])
		if structure[position[0]][position[1]+1] == 1:
			available.append(3) #check right
			q.append(Q_table[position[0]][position[1]][3])
		available = np.array(available)
		q = np.array(q)
		return available, q

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

class Action_Value_table():
	def	initialize(self):	
		return np.zeros((structure.shape[0], structure.shape[1], num_of_action))

	def update(self, prev_position, position, reward, action):
		td = reward + gamma*np.amax(Q_table[position[0]][position[1]]) - Q_table[prev_position[0]][prev_position[1]][action]
		Q_table[prev_position[0]][prev_position[1]][action] += alpha*td

agent = Agent()
Q = Action_Value_table()
Q_table = Q.initialize()
eval_log = []

def evaluation():
	total_step = 0
	for i in range(int(eval_times)):
		position = agent.initialize()
		while(True):
			action = agent.epsilon_greedy(position, Q_table, eval_epsilon)
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
		action = agent.epsilon_greedy(position, Q_table, epsilon)
		prev_position = copy.deepcopy(position)
		position = agent.move(action, position)
		reward = reward_table[position[0]][position[1]]
		Q.update(prev_position, position, reward, action)
		finish_flag = agent.finish(position)
		if finish_flag:
			break

#output
x = np.arange(n_episode)
y = np.array(eval_log)
plt.plot(x, y)
if big_maze == 1:
	plt.savefig('Q_big/curve(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
else:
	plt.savefig('Q/curve(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()	

fig,axn = plt.subplots(2, 2, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i, ax in enumerate(axn.flat):
	sns.heatmap(Q_table[:,:,i],linewidths=0.5, ax=ax,cbar=i == 0,vmin=0, vmax=1, cbar_ax=None if i else cbar_ax)
	ax.text(1.5, structure.shape[0]-1.5, "S", ha = 'center', va = 'center', size=8)
	ax.text(structure.shape[1]-1.5, 1.5 , "G", ha = 'center', va = 'center', size=8)
	if i == 0:
		ax.set_title("down")
	if i == 1:
		ax.set_title("up")
	if i == 2:
		ax.set_title("left")
	if i == 3:
		ax.set_title("right")	
fig.tight_layout(rect=[0, 0, .9, 1])
if big_maze == 1:
	plt.savefig('Q_big/Q_table(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
else:	
	plt.savefig('Q/Q_table(a={},g={},e={},n={}).png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()

sns.heatmap(structure, linewidths=0.5, cmap='Blues_r', cbar=False)
plt.text(1.5, structure.shape[0]-1.5, "S", ha = 'center', va = 'center')
plt.text(structure.shape[1]-1.5, 1.5 , "G", ha = 'center', va = 'center')
if big_maze == 1:
	plt.savefig('Q_big/maze.png'.format(alpha, gamma, epsilon, n_episode))
else:
	plt.savefig('Q/maze.png'.format(alpha, gamma, epsilon, n_episode))
plt.clf()





