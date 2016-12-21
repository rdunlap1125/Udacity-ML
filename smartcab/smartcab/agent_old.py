import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_hat = dict()
        self.alpha = 0.2    #learning rate
        self.gamma = 0.2    #discount
        self.epsilon = 0.1  #exploration rate

        self.stateCount = dict()
    
        for light in ['red', 'green']:
            for plannerRecommendation in Environment.valid_actions[1:]:
                for oncomingTraffic in [True, False]:
                    for leftTraffic in [True, False]:
                        for rightTraffic in [True, False]:
                            newDict = dict()
                            for action in Environment.valid_actions:
                                newDict[action] = 4.0
                            self.Q_hat[(light, oncomingTraffic, leftTraffic, rightTraffic, plannerRecommendation)] = newDict
                            self.stateCount[(light, oncomingTraffic, leftTraffic, rightTraffic, plannerRecommendation)] = 0

        # this code should be elsewhere based on SRP, but everything for submission has to be in agent.py
        self.current_trial = 0
        self.total_reward = 0.0         #total reward for a trial
        self.planner_penalties = 0.0    #total planner penalties for a trial
        self.traffic_penalties = 0.0    #total traffic-based penalties for a trial
        self.trial_succeeded = False


        self.simulation_trial_with_last_traffic_penalty = 0;
        self.simulation_trial_with_last_planner_penalty = 0;
        self.simulation_last_failed_trial = 0;
        self.simulation_trial_successes = 0;
        self.simulation_total_reward = 0.0      #total reward for a simulation run
        self.simulation_planner_penalties = 0.0 #total planner penalties for a simulation run
        self.simulation_traffic_penalties = 0.0 #total traffic penalties for a simulation run
        self.simulation_total_deadline = 0.0    #total of all deadlines for a simulation run
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.trial_succeeded:
            self.simulation_trial_successes += 1
        else:
            self.simulation_last_failed_trial = self.current_trial
        self.trial_succeeded = False
        self.current_trial += 1

        self.simulation_total_reward += self.total_reward
        self.simulation_planner_penalties += self.planner_penalties
        self.simulation_traffic_penalties += self.traffic_penalties
        
        self.total_reward = 0.0
        self.planner_penalties = 0.0
        self.traffic_penalties = 0.0
        
        self.simulation_total_deadline += self.env.get_deadline(self) #this data is best captured at the start of the trial
        
    def calcState(self, inputs, next_waypoint):
        state = (inputs['light'], \
                 False if inputs['oncoming'] == None else True, \
                 False if inputs['left'] == None else True, \
                 False if inputs['right'] == None else True, \
                 next_waypoint)
        return state
    
    def printBestActions(self):
          for light in ['red', 'green']:
            for plannerRecommendation in Environment.valid_actions[1:]:
                for oncomingTraffic in [True, False]:
                    for leftTraffic in [True, False]:
                        for rightTraffic in [True, False]:
                            state = (light, oncomingTraffic, leftTraffic, rightTraffic, plannerRecommendation)
                            action_dict = self.Q_hat[state]
                            action = max(action_dict, key=action_dict.get)
                            print state, action, self.stateCount[state]
            
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.calcState(inputs, self.next_waypoint)
        self.stateCount[self.state] += 1

        # TODO: Select action according to your policy   
        import random
        if random.random() < self.epsilon:
            action = random.choice(Environment.valid_actions)
        else:
            action_dict = self.Q_hat[self.state]
            action = max(action_dict, key=action_dict.get)        


        # Execute action and get reward
        reward = self.env.act(self, action)

        #The following metrics are not used for learning, but for evaluating the success of the agent
        #This tracking takes advantage of the fact that the penalties are mutually exclusive from each other
        #  and the positive rewards, since illegal traffic actions are not executed and you can never get a
        # planner penalty the same turn you reach a goal
        self.total_reward += reward
        if -0.7 < reward < -0.2:
            self.planner_penalties += reward
            self.simulation_trial_with_last_planner_penalty = self.current_trial
        if reward < -0.7:
            self.traffic_penalties += reward
            self.simulation_trial_with_last_traffic_penalty = self.current_trial            

        # TODO: Learn policy based on state, action, reward
        next_waypoint = self.planner.next_waypoint()
        if next_waypoint == None:
            futureUtility = 0.0 #we're done, so there is no next action
            self.trial_succeeded = True
        else:            
            new_state = self.calcState(self.env.sense(self), next_waypoint)
            futureUtility = max(self.Q_hat[new_state].values())
        self.Q_hat[self.state][action] = (1 - self.alpha) * self.Q_hat[self.state][action] \
                               + self.alpha * (reward + self.gamma * futureUtility)

##        #print "LearningAgent.update(): deadline = {}, inputs = {}, state = {}, action = {}, reward = {}".format(deadline, inputs, self.state, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    CONST_SIMULATIONS_PER_PARAMETER_SET = 1
    
    import numpy as np
    for candidate_alpha in np.linspace(0.2, 0.2, 1):
        for candidate_epsilon in np.linspace(0.1, 0.1, 1):
            for candidate_gamma in np.linspace(0.2, 0.2, 1):
                sum_deadlines = 0.0
                sum_reward = 0.0
                sum_planner_penalties = 0.0
                sum_traffic_penalties = 0.0
                for i in range(0, CONST_SIMULATIONS_PER_PARAMETER_SET):
                    # Set up environment and agent
                    e = Environment()  # create environment (also adds some dummy traffic)
                    a = e.create_agent(LearningAgent)  # create agent
                    a.alpha = candidate_alpha
                    a.epsilon = candidate_epsilon
                    a.gamma = candidate_gamma
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                    # Now simulate it
                    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
                    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                    sim.run(n_trials=100)  # run for a specified number of trials
                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    print a.current_trial
                    a.reset() #saves the results of the last trial

                    sum_deadlines += a.simulation_total_deadline
                    sum_reward += a.simulation_total_reward
                    sum_planner_penalties += a.simulation_planner_penalties
                    sum_traffic_penalties += a.simulation_traffic_penalties
                    a.printBestActions()

            print "alpha = {}; epsilon = {}; gamma = {}; reward ratio = {}".format(candidate_alpha, candidate_epsilon, candidate_gamma, sum_reward / sum_deadlines)
            
            
        print "Total deadline = {}".format(sum_deadlines)
        print "Total reward = {}; total planner penalties = {}; total traffic penalties = {}".format( \
            sum_reward, sum_planner_penalties, sum_traffic_penalties)
        print "Last planner penalty at {}; last traffic penalty at {}".format(a.simulation_trial_with_last_planner_penalty, a.simulation_trial_with_last_traffic_penalty)
        print "Total successes = {}; last failure at trial {}".format(a.simulation_trial_successes, a.simulation_last_failed_trial)

if __name__ == '__main__':
    run()
