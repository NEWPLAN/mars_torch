

class Trainer():
    def __init__(self, agent, env, config, record=True):
        self.env = env
        self.agent = agent
        self.config = config['args']
        pass

    def train(self, start_episode=0, start_step=0):

        for episode in range(start_episode, self.config.EPISODE):
            #TODO: env.reset()
            s0 = self.env.reset()
            episode_reward = 0

            for step in range(start_step, self.config.MAX_STEP):
                # env.render()
                a0 = self.agent.act(s0)
                s1, r1, done, _ = self.env.step(a0)
                self.agent.put(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self.agent.learn()
            start_step = 0

            # TODO: save checkpoint
            if self.config.CHECK_POINT_INTERVAL > 0 and episode % self.config.CHECK_POINT_INTERVAL == 0:
                print("Saving checkpoint at episode:", episode)
                self.agent.save_checkpoint(
                    episode, step, self.config.CHECKPOINT_DIR)

            print(episode, ': ', episode_reward)
        self.agent.save_model(self.config.OUTPUT_DIR)
