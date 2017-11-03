from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import seaborn as sns
import matplotlib.pyplot as plt


class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.q = {}
        self.alpha = 0.2
        self.discount = 0.1
        self.frameCounter = 0
        self.results = []
        self.ploty = []
        self.plotx = []
        self.highestScore = 0
        self.averageScore = 0
        self.totalScore = 0
        return

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def discretise(self, state):
        return (
            int(state['player_y']/(450/15)),
            int(state['player_vel']/(19/15)),
            int(state['next_pipe_top_y']/(250/15)),
            int(state['next_pipe_dist_to_player']/(280/15))
        )

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        s1 = self.discretise(s1)
        s2 = self.discretise(s2)
        if s1 not in self.q.keys():
            self.q.update({s1: {0: 0, 1: 0}})
        if s2 not in self.q.keys():
            self.q.update({s2: {0: 0, 1: 0}})
        #print(self.q[s1], self.q[s2])
        b = 1
        if a == 1:
            b = 0
        qval0 = (self.q[s1][a]*(1-self.alpha)) + (self.alpha*(r + self.discount*max(self.q[s2][0],self.q[s2][1])))
        qval1 = (self.q[s1][b])
        self.q.update({s1: {a: qval0, b: qval1}})
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        state = self.discretise(state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        greedy = True
        greed = random.randint(1, 100)
        if (greed <= 10):
            greedy = False
        action = random.randint(0, 1)
        if state not in self.q.keys():
            self.q.update({state: {0: 0, 1: 0}})
        if greedy:
            if self.q[state][0] < self.q[state][1]:
                action = 1
                #print(self.q[state], "wubbalubbadubdub scoobydoo longfuckingwords", self.q[state], state)
            elif self.q[state][0] > self.q[state][1]:
                action = 0
                #print(self.q[state])
        return action

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # print("state: %s" % state)
        state = agent.discretise(state)
        action = random.randint(0, 1)
        if state not in self.q.keys():
            return action
        if self.q[state][0] < self.q[state][1]:
            action = 1
        elif self.q[state][0] > self.q[state][1]:
            action = 0
        return action


def train_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        previousState = env.game.getGameState()
        action = agent.training_policy(previousState)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        agent.observe(previousState, action, reward, env.game.getGameState(), env.game_over())

        score += reward
        agent.results.append(score)
        agent.frameCounter += 1
        if agent.frameCounter % 10000 == 0 and agent.frameCounter != 0:
            sum = 0
            for x in range(len(agent.results)):
                sum += agent.results[x]
            avg = sum / 10000
            agent.ploty.append(avg)
            agent.plotx.append(agent.frameCounter)
            agent.results = []
        # reset the environment if the game is over
        if env.game_over():
            env.reset_game()
            nb_episodes -= 1
            score = 0

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
              reward_values=reward_values)
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print(score, nb_episodes)
            env.reset_game()
            nb_episodes -= 1
            if score > agent.highestScore:
                agent.highestScore = score
            agent.totalScore += score
            score = 0

agent = FlappyAgent()
for x in range (30):
    train_game(250, agent)
    agent.alpha -= agent.alpha/4
    agent.discount += (1-agent.discount)/2
    print(agent.alpha)
run_game(50, agent)
plt.show(sns.barplot(x=agent.plotx, y=agent.ploty))