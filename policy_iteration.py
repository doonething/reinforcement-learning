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
   for i in range( 14) :
   #i=1 
       for j in range( 4 ) :
	      random_policy [i,j] =0


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA 
    set_policy_to_zero(policy) 
    print(" before Policy Probability Distribution:") 
    print(policy) 
    print("")
   	
    while True:
        # Evaluate the current policy
       	
       	
        V = policy_eval_fn(policy, env, discount_factor) 
        print 
        print ' Evaluated Values ' 
        print V 
        print 
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
#            print 'policy[s] = ', policy[s] 
#             print 'policy[s][0] = ', policy[s][0] 
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V



random_policy = np.ones([env.nS, env.nA]) / env.nA




policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

# k
exit()

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")



