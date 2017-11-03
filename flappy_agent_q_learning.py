from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import seaborn as sns
import matplotlib.pyplot as plt


class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        self.q = {}
        self.alpha = 0.1
        self.discount = 0.95
        self.frameCounter = 0
        self.results = []
        self.ploty = []
        self.plotx = []
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
            int(state['player_y'] / 512 * 15.0),
            int(state['player_vel']),
            int(state['next_pipe_top_y'] / 512 * 15.0),
            int(state['next_pipe_dist_to_player'] / 512 * 15.0)
        )

    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        if s1 not in self.q.keys():
            self.q.update({s1: {0: 0, 1: 0}})
        if s2 not in self.q.keys():
            self.q.update({s2: {0: 0, 1: 0}})
        qval0 = (self.q[s1][0]*(1-self.alpha)) + (self.alpha*(r + self.discount*max(self.q[s2][0],self.q[s2][1])))
        qval1 = (self.q[s1][1]*(1-self.alpha)) + (self.alpha*(r + self.discount*max(self.q[s2][0],self.q[s2][1])))
        self.q.update({s1: {0: qval0, 1: qval1}})
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        # print("state: %s" % state)
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
            elif self.q[state][0] > self.q[state][1]:
                action = 0
        return action

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
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

    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    # display_screen=False, force_fps=True
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        previousState = agent.discretise(env.game.getGameState())
        action = agent.training_policy(previousState)
        # step the environment
        reward = env.act(env.getActionSet()[action])
        # TODO: for training let the agent observe the current state transition
        agent.observe(previousState, action, reward, agent.discretise(env.game.getGameState()), env.game_over())

        score += reward
        agent.results.append(score)
        agent.frameCounter += 1
        if agent.frameCounter % 5000 == 0 and agent.frameCounter != 0:
            sum = 0
            for x in range(len(agent.results)):
                sum += agent.results[x]
            avg = sum / 5000
            agent.ploty.append(avg)
            agent.plotx.append(agent.frameCounter)
            agent.results = []
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True
    env.init()

    score = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(env.game.getGameState())

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0


agent = FlappyAgent()
train_game(30000, agent)
run_game(20, agent)
plt.show(sns.barplot(x=agent.plotx, y=agent.ploty))