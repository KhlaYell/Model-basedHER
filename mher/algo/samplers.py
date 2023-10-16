from mher.common import logger
import numpy as np


def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]  # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]  # number of episodes
        batch_size = batch_size_in_transitions  # number of goals sample from rollout
        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _random_sample


def obs_to_goal_fun(env):
    from gym.envs.robotics import FetchEnv, hand_env
    from mher.envs import point2d
    from mher.envs import sawyer_reach
    from gym.envs.mujoco import reacher
    from mher.envs.sawyer_door_hook import SawyerDoorHookEnv

    tmp_env = env
    while hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env

    if isinstance(tmp_env, FetchEnv):
        obs_dim = env.observation_space['observation'].shape[0]
        goal_dim = env.observation_space['desired_goal'].shape[0]
        temp_dim = env.sim.data.get_site_xpos('robot0:grip').shape[0]

        def obs_to_goal(observation):
            observation = observation.reshape(-1, obs_dim)
            if env.has_object:
                goal = observation[:, temp_dim:temp_dim + goal_dim]
            else:
                goal = observation[:, :goal_dim]
            return goal.copy()
    elif isinstance(tmp_env, hand_env.HandEnv):
        goal_dim = env.observation_space['desired_goal'].shape[0]

        def obs_to_goal(observation):
            goal = observation[:, -goal_dim:]
            return goal.copy()
    elif isinstance(tmp_env, point2d.Point2DEnv):
        def obs_to_goal(observation):
            return observation.copy()
    elif isinstance(tmp_env, sawyer_reach.SawyerReachXYZEnv):
        def obs_to_goal(observation):
            return observation
    elif isinstance(tmp_env, reacher.ReacherEnv):
        def obs_to_goal(observation):
            return observation[:, -3:-1]
    elif isinstance(tmp_env, SawyerDoorHookEnv):
        def obs_to_goal(observation):
            return observation[:, -1]
    else:
        raise NotImplementedError('Do not support such type {}'.format(env))

    return obs_to_goal


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, obs_to_goal_fun=None, no_her=False):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    if no_her:
        print('*' * 10 + 'Do not use HER in this method' + '*' * 10)

    def _random_log(string):
        if np.random.random() < 0.02:
            print(string)

    def _preprocess(episode_batch, batch_size_in_transitions, n):
        T = episode_batch[f'o_{(n + 1)}'].shape[1]  # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]  # number of episodes
        batch_size = batch_size_in_transitions  # number of goals sample from rollout

        # Select which episodes and time steps to use.
        # np.random.randint doesn't contain the last one, so comes from 0 to roolout_batch_size-1
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        # Reconstruct info dictionary for reward  computation.
        info = {}
        reward_params = {'ag_2': ag_2, 'g': g}
        reward_params['info'] = info
        return reward_fun(**reward_params)

    def _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        return future_ag.copy(), her_indexes.copy()

    def _get_ags_from_states(batch_size, states, ratio=0.8, indexs=None):
        if indexs is None:
            indexs = (np.random.uniform(size=batch_size) < ratio)
        next_goals = obs_to_goal_fun(states[indexs])
        return next_goals.copy(), indexs.copy()

    def _reshape_transitions(transitions, batch_size):
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size)
        return transitions

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, n, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, n)
        if not no_her:
            future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size)

    def _dynamic_interaction_full(o, g, action_fun, dynamic_model, steps, n):
        batch_size = o.shape[0]
        last_state = o.copy()
        states_list, actions_list, next_states_list = [], [], []
        goals_list, ags_list, next_ags_list, reward_list = [], [], [], []
        for _ in range(0, steps):
            goals_list.append(g.copy())
            states_list.append(last_state.copy())
            ag_array = obs_to_goal_fun(last_state).copy()
            ags_list.append(ag_array)

            action_array = action_fun(o=last_state, g=g)
            action_array += 0.2 * np.random.randn(*action_array.shape)  # gaussian noise
            action_array = np.clip(action_array, -1, 1)
            next_state_array = dynamic_model.predict_next_state(last_state, action_array)

            actions_list.append(action_array.copy())
            next_states_list.append(next_state_array.copy())
            next_ag_array = obs_to_goal_fun(next_state_array).copy()
            next_ags_list.append(next_ag_array)
            reward_list.append(_get_reward(next_ag_array, g))
            last_state = next_state_array
        transitions = {}
        transitions['o'] = np.concatenate(states_list, axis=0).reshape(batch_size * steps, -1)
        transitions['ag'] = np.concatenate(ags_list, axis=0).reshape(batch_size * steps, -1)
        for i in range(2, (n + 2)):
            transitions[f'o_{i}'] = np.concatenate(next_states_list, axis=0).reshape(batch_size * steps, -1)
            transitions[f'ag_{i}'] = np.concatenate(next_ags_list, axis=0).reshape(batch_size * steps, -1)
        transitions['g'] = np.concatenate(goals_list, axis=0).reshape(batch_size * steps, -1)
        transitions['r'] = np.concatenate(reward_list, axis=0).reshape(batch_size * steps, -1)
        transitions['u'] = np.concatenate(actions_list, axis=0).reshape(batch_size * steps, -1)
        return transitions

    def _dynamic_interaction(o, g, action_fun, dynamic_model, steps, transitions, n, act_noise=0):
        last_state = o.copy()
        next_states_list = []
        for _ in range(0, steps):
            action_array = action_fun(o=last_state, g=g)
            if act_noise > 0:  # action noise
                action_array += np.random.normal(scale=act_noise, size=action_array.shape)
                action_array = np.clip(action_array, -1, 1)

            next_state_array = dynamic_model.predict_next_state(last_state, action_array, transitions, n)
            next_states_list.append(next_state_array.copy())
            last_state = next_state_array
        return next_states_list

    def _sample_mve_transitions(episode_batch, batch_size_in_transitions, n, info):
        steps, gamma, Q_fun = info['nstep'], info['gamma'], info['get_Q_pi']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, n)

        _random_log('using goal mve sampler with step:{}'.format(steps))
        # update dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions, n, times=2)

        last_state = transitions['o_2'].copy()
        next_states_list = _dynamic_interaction(last_state, transitions['g'], action_fun, dynamic_model, steps - 1)
        next_states_list.insert(0, last_state.copy())
        reward_lis = []
        for next_state in next_states_list:
            reward = _get_reward(obs_to_goal_fun(next_state), transitions['g'])
            reward_lis.append(reward.copy())
        target = 0
        for i in range(0, steps):
            target += pow(gamma, i) * reward_lis[i]
        target += pow(gamma, steps) * Q_fun(next_states_list[-1], transitions['g']).reshape(-1)
        transitions['r'] = target.copy()
        return _reshape_transitions(transitions, batch_size)

    def _sample_mbpo_transitions(episode_batch, batch_size_in_transitions, n, info):
        dynamic_model, action_fun, steps = info['dynamic_model'], info['action_fun'], info['nstep']
        model_samples_buffer = info['model_buffer']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, n)

        _random_log('using goal mbpo sampler with step:{}'.format(steps))
        # update dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions, n, times=2)
        model_transitions = _dynamic_interaction_full(transitions['o'], transitions['g'], action_fun, dynamic_model,
                                                      steps)
        model_samples_buffer.store_transitions(model_transitions)
        sample_model_batches = model_samples_buffer.sample(batch_size)
        return _reshape_transitions(sample_model_batches, batch_size)

    def _sample_nstep_dynamic_her_transitions(episode_batch, batch_size_in_transitions, n, info):
        steps, gamma, Q_fun, alpha = info['nstep'], info['gamma'], info['get_Q_pi'], info['alpha']
        dynamic_model, action_fun = info['dynamic_model'], info['action_fun']
        dynamic_ag_ratio = info['mb_relabeling_ratio']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions, n)
        train_policy, no_mb_relabel, no_mgsl = info['train_policy'], info['no_mb_relabel'], info['no_mgsl']
        dynamic_ag_ratio_cur = dynamic_ag_ratio

        _random_log('using mher with step:{}, alpha:{}, and dynamic relabeling rate:{}'.format(steps, alpha,
                                                                                               dynamic_ag_ratio_cur))
        # update dynamic model
        loss = dynamic_model.update(transitions['o'], transitions['u'], transitions, n, times=2)

        relabel_indexes = (np.random.uniform(size=batch_size) < dynamic_ag_ratio_cur)
        # # Re-compute reward since we may have substituted the goal.
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        # model-based relabeling
        last_state = transitions['o_2'].copy()
        if dynamic_ag_ratio_cur > 0:
            next_states_list = _dynamic_interaction(last_state, transitions['g'], action_fun, dynamic_model, steps,
                                                    transitions, n, act_noise=0.2)
            next_states_list.insert(0, last_state.copy())
            next_states_array = np.concatenate(next_states_list, axis=1).reshape(batch_size, steps + 1, -1)
            step_idx = np.random.randint(next_states_array.shape[1], size=(batch_size))
            last_state = next_states_array[np.arange(batch_size).reshape(-1), step_idx]
            # add dynamic achieve goals
            new_ags, _ = _get_ags_from_states(batch_size, last_state, 1)

            if not no_mb_relabel:
                transitions['g'][relabel_indexes] = new_ags[relabel_indexes].reshape((relabel_indexes.sum(), -1))

            transitions['idxs'] = relabel_indexes.copy()
            if not no_mgsl and no_mb_relabel:
                # Auxilary task for no MBR (set alpha=0)
                train_policy(o=transitions['o'][relabel_indexes], g=new_ags[relabel_indexes],
                             u=transitions['u'][relabel_indexes])

                # recompute rewards
            transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])

        target_step1 = transitions['r'] + gamma * Q_fun(o=transitions['o_2'], g=transitions['g']).reshape(-1)
        transitions['r'] = target_step1.copy()
        return _reshape_transitions(transitions, batch_size)

    def _sample_nstep_supervised_her_transitions(episode_batch, batch_size_in_transitions, info):
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        train_policy = info['train_policy']

        _random_log('using nstep supervide policy learning')
        future_ag, her_indexes = _get_her_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1)
        transitions['g'][her_indexes] = future_ag
        loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'])
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])  # no need, but in order to unify
        return _reshape_transitions(transitions, batch_size)

    return _sample_her_transitions, _sample_nstep_dynamic_her_transitions, _sample_nstep_supervised_her_transitions, \
        _sample_mve_transitions, _sample_mbpo_transitions

