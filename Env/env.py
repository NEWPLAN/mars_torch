import random
import numpy as np

from log import logger


class NetEnv():
    def __init__(self, folder_path, TM_path, topo_path, step_per_episode, episode_start_index_=0):
        # for repeat experiment
        random.seed(666)
        np.random.seed(666)

        self.MAX_STEP_PER_EPISODE = step_per_episode

        self.tm_path_ = folder_path + TM_path + ".txt"
        self.topo_path_ = folder_path + topo_path + ".txt"

        self.tunnels_ = []  # a list to store all candidate path
        self.tunnel_num_ = 0  # all the candidate path num
        self.tunnel_num_4_one_pair = []
        self.tunnel_shape_ = []

        self.loads_on_tunnels_ = []  # TM loads on each candidate path
        self.adjacency_matrix_ = []  # Adjacency matrix
        self.capacity_matrix_ = []  # the bandwidth of each links
        self.link_num_ = 0

        self.node_num_ = 0  # node number

        # running time state
        self.flow_map_ = []  # the flow map
        self.step_count = 0  # use a global counter to record how many steps are operated
        # use a global counter to record how many episode are operated
        self.episode_count = episode_start_index_-1
        self.loads_on_tunnels_now_ = []  # the  TM loadings for using now

        # tmp variable
        self.lines = None
        self.tm_start_index = 0

        # loading file and init flow map
        self.load_file()
        self.init_flow_map()
        # exit(0)

    def load_file(self):
        logger.info("loading file")

        # lines = None
        # line_for_TM = 0

        def parsing_candidate_path():  # get all candidate paths
            logger.info("Parsing candidate path")
            candidate_path_tmp_ = []
            line_for_TM = 0
            for i in range(1, len(self.lines)):
                if self.lines[i].strip() == "succeed":
                    line_for_TM = i
                    break
                line_list = self.lines[i].strip().split(',')
                try:
                    if len(candidate_path_tmp_) != 0 and (int(line_list[1]) != candidate_path_tmp_[0][0] or int(line_list[-2]) != candidate_path_tmp_[0][-1]):
                        self.tunnels_.append(candidate_path_tmp_)
                        candidate_path_tmp_ = []
                    candidate_path_tmp_.append(list(map(int, line_list[1:-1])))
                except Exception as e:
                    print(e, line_list, candidate_path_tmp_)
                    exit(0)
            self.tunnels_.append(candidate_path_tmp_)

            self.tunnel_num_ = len(self.tunnels_)
            self.tunnel_num_4_one_pair = [len(item) for item in self.tunnels_]
            self.tm_start_index = line_for_TM

        def parsing_traffic_matrices():
            logger.info("Parsing traffic matrices")
            for tm_index in range(self.tm_start_index + 1, len(self.lines)):
                cp_rate_tmp = list(
                    map(float, self.lines[tm_index].strip().split(',')))
                self.loads_on_tunnels_.append(cp_rate_tmp)

                assert len(self.loads_on_tunnels_[0]) == len(
                    self.loads_on_tunnels_[-1])

        def get_node_num():
            for item in self.tunnels_:  # get the maximum node id as the node num
                self.node_num_ = max(
                    self.node_num_, max([max(i) for i in item]))
            self.node_num_ += 1
            # since the node index is from 0, return the the node num with self.node_num+=1

        def parsing_adjacency_matrix():
            logger.info("Parsing adjacency matrix")
            # init adjacency_matrix
            for row in range(self.node_num_):
                self.adjacency_matrix_.append([])
                for col in range(self.node_num_):
                    self.adjacency_matrix_[row].append(0)

            # parsing adjacency_matrix
            for i in range(self.tunnel_num_):
                for j in range(self.tunnel_num_4_one_pair[i]):
                    for k in range(len(self.tunnels_[i][j]) - 1):
                        src = self.tunnels_[i][j][k]
                        dst = self.tunnels_[i][j][k + 1]
                        self.adjacency_matrix_[src][dst] = 1

            for row in range(self.node_num_):
                for col in range(self.node_num_):
                    self.link_num_ += self.adjacency_matrix_[row][col]

        def loading_all_file():
            logger.info("Loading TM files")
            with open(self.tm_path_, 'r') as f:
                self.lines = f.readlines()
            logger.info("Loading topology files")
            with open(self.topo_path_, 'r') as f:
                self.topo_lines = f.readlines()

        def parsing_candidate_path_and_TMs():
            parsing_candidate_path()
            parsing_traffic_matrices()
            get_node_num()
            parsing_adjacency_matrix()

            # build the decentralized agent
            self.tunnel_shape_ = [[] for x in range(self.node_num_)]
            for x in self.tunnels_:
                print(x, x[0][0])
                self.tunnel_shape_[x[0][0]].append(len(x))

        def parsing_topology():
            logger.info("Parsing topology")
            for index in range(self.node_num_):
                crow = [0] * self.node_num_
                self.capacity_matrix_.append(crow)

            for index in range(1, len(self.topo_lines)):
                lineList = self.topo_lines[index].strip().split(' ')
                src = int(lineList[0]) - 1
                dst = int(lineList[1]) - 1
                self.capacity_matrix_[src][dst] = float(lineList[3])
                self.capacity_matrix_[dst][src] = float(lineList[3])

        loading_all_file()
        parsing_candidate_path_and_TMs()
        parsing_topology()
        logger.info("Parsing done")
        # self.show_info()

    def init_flow_map(self):
        logger.info("init flow map")
        for row in range(self.node_num_):
            self.flow_map_.append([])
            for col in range(self.node_num_):
                self.flow_map_[row].append(0)

    def show_info(self):
        logger.info("Show the tunnels, i.e., candidate path")
        for x in self.tunnels_:
            print(x)
        logger.info(
            "the total tunnel number:{}, and tunnel number for one o-d pair:\n{}".format(self.tunnel_num_, self.tunnel_num_4_one_pair))

        logger.info("The node number: {}".format(self.node_num_))

        logger.info("The adjacency matrix:")
        print('   ', end='')
        for node_index in range(1, self.node_num_+1):
            print(node_index, end='  ' if node_index < 10 else ' ')
        print()
        node_index = 1
        for x in self.adjacency_matrix_:
            print(node_index, x)
            node_index += 1

        logger.info("The capacity matrix:")
        for item in self.capacity_matrix_:
            print(item)

    def update_flow_map(self, action):
        # if self.step_count % 100 == 0:
        #     logger.info("updating flow map for the step: {}:{}".format(
        #         self.episode_count, self.step_count))

        def preprocess_action(action):
            if action == []:  # use the default action, i.e., split evenly
                #logger.warning( "Your action is empty, reset to the default action")
                for item in self.tunnel_num_4_one_pair:
                    action += [round(1.0 / item, 4) for j in range(item)]
            return action

        def preprocess_tm_loading(action):
            sub_TM_loading = []
            # the TM is the total mount of traffic with O-D pair based
            count = 0
            # extract the TM loads according to the action, i.e., the split ratio
            for i in range(self.tunnel_num_):
                sub_TM_loading.append([])
                for j in range(self.tunnel_num_4_one_pair[i]):
                    tmp = 0
                    if j == self.tunnel_num_4_one_pair[i] - 1:
                        tmp = self.loads_on_tunnels_now_[
                            i] - sum(sub_TM_loading[i])
                    else:
                        tmp = self.loads_on_tunnels_now_[i] * action[count]
                    count += 1
                    sub_TM_loading[i].append(tmp)
            return sub_TM_loading

        def assigning_flow(sub_TM_loading):
            for i in range(self.node_num_):
                for j in range(self.node_num_):
                    self.flow_map_[i][j] = 0

            for i in range(self.tunnel_num_):
                for j in range(self.tunnel_num_4_one_pair[i]):
                    # logger.info("{}:{}\n{}".format(i, j, self.tunnels_[i][j]))
                    for k in range(len(self.tunnels_[i][j]) - 1):
                        # update loads for each link
                        src = self.tunnels_[i][j][k]
                        dst = self.tunnels_[i][j][k + 1]
                        self.flow_map_[src][dst] += sub_TM_loading[i][j]

        action_ = preprocess_action(action)
        sub_TM_loading_ = preprocess_tm_loading(action_)
        assigning_flow(sub_TM_loading_)

        # for x in self.flow_map_:
        #     print(x)
        # exit(0)

    def parsing_state_at_runtime(self):
        # logger.info("Parsing the network utils at runtime")

        # parsing each tunnel utils
        tunnel_util_ = []
        for i in range(self.tunnel_num_):
            tunnel_util_.append([])  # use a [] to seperate each src
            for j in range(self.tunnel_num_4_one_pair[i]):
                path_util = []  # path for each src-dst pair:
                for k in range(len(self.tunnels_[i][j]) - 1):
                    src = self.tunnels_[i][j][k]
                    dst = self.tunnels_[i][j][k + 1]
                    path_util.append(
                        round(self.flow_map_[src][dst] / self.capacity_matrix_[src][dst], 4))
                tunnel_util_.append(path_util)

        # parsing link utils
        link_util_ = []
        link_ID_ = []
        for i in range(self.node_num_):
            for j in range(self.node_num_):
                # here we just observe the connected link
                if self.adjacency_matrix_[i][j] != 0:
                    link_ID_.append([i, j])
                    link_util_.append(
                        round(self.flow_map_[i][j]/self.capacity_matrix_[i][j], 4))

        max_util_ = max(link_util_)  # the maximum link utilization
        # print(tunnel_util_)
        # print(link_ID_)
        # print(link_util_)
        # print(max_util_)
        return round(max_util_, 4), tunnel_util_, link_util_

    def loading_TMs(self, index=0):
        TM_num = len(self.loads_on_tunnels_)
        logger.info("Loading the {}/{} TM".format(index, TM_num))
        self.episode_count = self.episode_count % TM_num
        self.loads_on_tunnels_now_ = self.loads_on_tunnels_[self.episode_count]
        print(self.loads_on_tunnels_now_)

    def step(self, action):

        if self.step_count and self.step_count % self.MAX_STEP_PER_EPISODE == 0:  # reload new TMs
            self.episode_count += 1
            self.loading_TMs(self.episode_count)
            self.step_count = 0
            assert 1 == 0, "never come here"

        # logger.info("In step {}:{}".format( self.episode_count, self.step_count))
        self.update_flow_map(action=[])
        max_util, tunnel_util, link_util = self.parsing_state_at_runtime()
        # if self.step_count == 0:
        #     logger.info("{}".format(max_util))
        self.step_count += 1
        return max_util, tunnel_util, link_util

    def reset(self, episode_start_index=0):
        logger.info("Reset the env")

        self.episode_count = episode_start_index
        self.loading_TMs(self.episode_count)
        self.step_count = 0

        self.update_flow_map(action=[])
        max_util, tunnel_util, link_util = self.parsing_state_at_runtime()
        # tunnel_split_by_src = [len(tunnel_util[i])
        #                        for i in range(len(tunnel_util))]
        # print(tunnel_split_by_src)
        return max_util, tunnel_util, link_util

    def get_state_dim(self, decentralized=0):
        if not decentralized:
            # print("link nums: ", self.link_num_)
            return self.link_num_

        return self.tunnel_shape_
        pass

    def get_action_dim(self, decentralized=0):

        if not decentralized:
            # for x in self.tunnel_shape_:
            #     print(x)
            return self.tunnel_num_4_one_pair
        print(self.tunnel_num_)
        pass
