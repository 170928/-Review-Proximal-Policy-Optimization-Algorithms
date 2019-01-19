import tensorflow as tf
import copy
from Default.network_ppo import network
from mlagents.envs import UnityEnvironment
import numpy as np

action_size = 5
vector_state_size = 6 * 3
visual_state_size = [84, 84, 3]

class PPOTrain:
    def __init__(self, ppo, old_ppo, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):

        self.ppo = ppo
        self.old_ppo = old_ppo
        self.gamma = gamma

        pi_trainable = self.ppo.get_trainable_variables()
        old_pi_trainable = self.old_ppo.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

        act_probs = self.ppo.act_probs
        act_probs_old = self.old_ppo.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss/clip'):
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(act_probs + 1e-10) - tf.log(act_probs_old + 1e-10))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)

        # construct computation graph for loss of value function
        with tf.variable_scope('loss/vf'):
            v_preds = self.ppo.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

        # construct computation graph for loss of entropy bonus
        with tf.variable_scope('loss/entropy'):
            entropy = - tf.reduce_sum(self.ppo.act_probs * tf.log(tf.clip_by_value(self.ppo.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)

        with tf.variable_scope('loss'):
            loss = loss_clip - c_1 * loss_vf
            loss = -loss
            self.loss = loss

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)


    def train(self, vecObs, visObs, actions, rewards, v_preds_next, gaes, sess):
        #self.vector_obs = tf.placeholder(dtype=tf.float32, shape=[None, vector_state_size])
        #self.visual_obs = tf.placeholder(dtype=tf.float32, shape=[None, visual_state_size[0], visual_state_size[1], visual_state_size[2]])

        return sess.run([self.loss, self.train_op], feed_dict={self.ppo.vector_obs: vecObs,
                                                                 self.ppo.visual_obs: visObs,
                                                                 self.old_ppo.vector_obs: vecObs,
                                                                 self.old_ppo.visual_obs: visObs,
                                                                 self.actions: actions,
                                                                 self.rewards: rewards,
                                                                 self.v_preds_next: v_preds_next,
                                                                 self.gaes: gaes})

    def assign_policy_parameters(self, sess):
        # assign policy parameter values to old policy parameters
        return sess.run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

if __name__ == '__main__':

    batch_size = 128
    mem_maxlen = 10000
    discount_factor = 0.99
    learning_rate = 0.00025

    run_episode = 20000
    test_episode = 50

    start_train_episode = 10

    target_update_step = 10000
    print_interval = 10
    save_interval = 10

    epsilon = 0.9
    epsilon_min = 0.1

    env_name =
    save_path =
    load_path =

    env = UnityEnvironment(file_name=env_name, worker_id=0)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    onPolicy = network('online')
    targetPolicy = network('target')
    agent = PPOTrain(onPolicy, targetPolicy)

    train_mode = True
    load_model = False

    env_info = env.reset(train_mode=train_mode)[default_brain]

    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    load_path = load_path
    # self.Summary,self.Merge = self.make_Summary()
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess = tf.Session(config=config)

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(load_path)
        Saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restore Model")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialize Model")

    # Reset Environment =======================
    env_info = env.reset(train_mode=train_mode)
    print("Env Reset")
    # =========================================

    mean_rewards = []

    # [차원 정보]
    # actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
    # rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
    # v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
    # gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')

    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False

        vector_states = []
        visual_states = []
        actions = []
        v_preds = []
        rewards = []

        env_info = env.reset(train_mode=train_mode)[default_brain]

        vector_state = env_info.vector_observations[0]
        visual_state = np.uint8(255 * env_info.visual_observations[0])


        episode_rewards = 0
        done = False

        while not done:
            vector_state = np.reshape(vector_state, newshape=(-1,vector_state_size))
            act_probs, action, v_pred = onPolicy.act(vecObs = vector_state, visObs = visual_state, sess = sess)

            if train_mode == True and epsilon > np.random.rand():
                action = [np.random.randint(1, action_size)]
            else:
                action = action
            env_info = env.step(action)[default_brain]

            next_vector_state = env_info.vector_observations[0]
            next_visual_state = np.uint8(255 * env_info.visual_observations[0])
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]


            # Reshape
            vector_state = np.reshape(vector_state, newshape=(-1, vector_state_size))
            visual_state = np.reshape(visual_state, newshape=(-1, visual_state_size[0], visual_state_size[1], visual_state_size[2]))
            action = np.reshape(action, newshape=(-1,))
            v_pred = np.reshape(v_pred, newshape=(-1,))
            reward = np.reshape(reward, newshape=(-1,))

            # Buffer
            vector_states.append(vector_state)
            visual_states.append(visual_state)
            actions.append(action)
            v_preds.append(v_pred)
            rewards.append(reward)

            if done:
                if train_mode and epsilon > epsilon_min:
                    epsilon -= 1 / run_episode

                v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                mean_rewards.append(episode_rewards)
                break
            else:
                vector_state = next_vector_state
                visual_state = next_visual_state

        # Calculate Gaes And Training.
        # ========================================================================================================


        gaes = agent.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

        # convert list to numpy array for feeding tf.placeholder
        vector_states = np.reshape(vector_states, newshape=(-1, vector_state_size))
        visual_states = np.reshape(visual_states, newshape=(-1, visual_state_size[0], visual_state_size[1], visual_state_size[2]))
        actions = np.reshape(actions, newshape=(-1,)).astype(dtype=np.int32)
        rewards = np.reshape(rewards, newshape=(-1,)).astype(dtype=np.float32)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
        gaes = np.reshape(gaes, newshape=(-1, )).astype(dtype=np.float32)
        gaes = (gaes - gaes.mean()) / gaes.std()

        agent.assign_policy_parameters(sess)

        inp = [vector_states, visual_states, actions, rewards, v_preds_next, gaes]

        # train
        losses = []
        for epoch in range(5):
            sample_indices = np.random.randint(low=0, high=vector_states.shape[0], size=batch_size)  # indices are in [low, high)
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
            loss, _ = agent.train(vecObs=sampled_inp[0],
                                    visObs=sampled_inp[1],
                                    actions=sampled_inp[2],
                                    rewards=sampled_inp[3],
                                    v_preds_next=sampled_inp[4],
                                    gaes=sampled_inp[5],
                                    sess=sess)
            losses.append(loss)


        if episode % print_interval == 0 and episode != 0:
            print("episode: {} / reward: {:.2f} / loss: {} / epsilon {}".format
                  (episode, np.mean(mean_rewards), np.mean(losses), epsilon))
            mean_rewards = []
            losses = []

        if episode % save_interval == 0 and episode != 0:
            Saver.save(sess, save_path + "/model.ckpt")
            print("Save Model {}".format(episode))
