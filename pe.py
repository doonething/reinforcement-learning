import numpy as np
import pprint
import sys
if "../" not in sys.path:
	sys.path.append("../") 
	from lib.envs.gridworld import GridworldEnv


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
	"""
	Evaluate a policy given an environment and a full description of the environment's dynamics.

	Args:
	policy: [S, A] shaped matrix representing the policy.
	env: OpenAI env. env.P represents the transition probabilities of the environment.
		env.P[s][a] is a (prob, next_state, reward, done) tuple.
	theta: We stop evaluation one our value function change is less than theta for all states.
	discount_factor: lambda discount factor.

	Returns:
	Vector of length env.nS representing the value function.
	"""
	#
	# k
	cnt = 0
	#
	# Start with a random (all 0) value function
	V = np.zeros(env.nS)
	while True:
		delta = 0
		# For each state, perform a "full backup"
		for s in range(env.nS):
			v = 0
			# Look at the possible next actions
			for a, action_prob in enumerate(policy[s]):
				# For each action, look at the possible next states...
				for  prob, next_state, reward, done in env.P[s][a]:
					# Calculate the expected value
					v += action_prob * prob * (reward + discount_factor * V[next_state])
			# How much our value function changed (across any states)
			delta = max(delta, np.abs(v - V[s]))
			#
			# k 
			if cnt % 100 == 0 :	
				print 'cnt=%5d delta = %3.3f v=%3.3f'% ( cnt, delta, v )
			cnt += 1
			#
			V[s] = v
			# Stop evaluating once our value function change is below a threshold
		if delta < theta:
			print 'ok'
			break
	return np.array(V)

# k
def set_policy_to_zero( random_policy ) :
		for i in range( 16 ) :
			for j in range( 4 ) :
				random_policy [i,j] =0

random_policy = np.ones([env.nS, env.nA]) / env.nA

# k
set_policy_to_zero(random_policy)

v = policy_eval(random_policy, env)

# k
exit()

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")



