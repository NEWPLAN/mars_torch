

from log import logger


class Trainer():
    def __init__(self, agent, env, config, record=True):
        self.env = env
        self.agent = agent
        self.config = config['args']

        self.env_network = config['env_network']
        pass

    def train(self, start_episode=0, start_step=0):

        for episode in range(start_episode, self.config.EPISODE):
            #TODO: env.reset()
            s0 = self.env.reset()
            u_max, tunnel_util, link_util = self.env_network.reset(
                episode)  # back up for network env
            episode_reward = 0
            logger.info("Max util: {}".format(u_max))

            for step in range(start_step, self.config.MAX_STEP):
                # env.render()
                a0 = self.agent.act(s0)
                s1, r1, done, _ = self.env.step(a0)
                # u_max, tunnel_util, link_util = self.env_network.step(
                #     a0)  # back up for network env
                # data_ = self.agent.act_backup(link_util)
                # critic_data_ = self.agent.crit_backup(link_util, data_)
                # print("Actor: ", data_)
                # print("Critic: ", critic_data_)
                # exit(0)
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
