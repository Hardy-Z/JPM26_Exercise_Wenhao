from config import *
from utils import *
import models
from tqdm import tqdm


def train_lifetime(show_plot = False):
    """Trian a policy network to maximize the life-time reward

    Args:
        show_plot (bool, optional): if show_plot = True, it will show a plot after training process. Defaults to False.

    Returns:
        losses, Euler_residuals, expected_rewards, rewards_
    """
    policy_net_liftime = models.PolicyNet()
    test_policy_expected(policy_net_liftime, test_rounds=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE_1)

    @tf.function
    def training_step():
        with tf.GradientTape() as tape:
            loss = -expected_lifetime_reward(policy_net = policy_net_liftime, T = Config.T) # type: ignore

        grads = tape.gradient(loss, policy_net_liftime.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net_liftime.trainable_variables))
        return loss

    # List to store loss and reward during training progress
    losses = []
    expected_rewards = []
    rewards = []
    Euler_residuals = []

    for epoch in tqdm(range(Config.EPOCHS_1), desc="Training Progress (maximize lifetime reward)"):

        expected_reward_value = expected_lifetime_reward(policy_net = policy_net_liftime)
        reward_value = lifetime_reward(policy_net = policy_net_liftime)
        Euler_residual_value = Euler_residual(policy_net = policy_net_liftime)
        loss_value = training_step()

        expected_rewards.append(expected_reward_value)
        rewards.append(reward_value)
        losses.append(loss_value)
        Euler_residuals.append(Euler_residual_value)

    if (show_plot == True):
        print("Showing plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.plot(rewards, label="Lifetime Reward")
        plt.plot(expected_rewards, label="Expected Lifetime Reward")
        plt.title("Loss over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rewawrd")
        plt.grid()
        plt.show()
    return losses, Euler_residuals, expected_rewards, rewards


def train_Euler(show_plot = False):
    """Trian a policy network to minimize the Euler-residual

    Args:
        show_plot (bool, optional): if show_plot = True, it will show a plot after training process. Defaults to False.

    Returns:
        losses, Euler_residuals, expected_rewards, rewards
    """
    policy_net_Euler = models.PolicyNet()
    test_policy_expected(policy_net_Euler, test_rounds=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE_2)

    @tf.function
    def training_step_Euler():
        with tf.GradientTape() as tape:
            loss = Euler_residual(policy_net = policy_net_Euler) 

        grads = tape.gradient(loss, policy_net_Euler.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net_Euler.trainable_variables))
        return loss

    # List to store loss and reward during training progress
    losses = []
    expected_rewards = []
    rewards = []
    Euler_residuals = []

    for epoch in tqdm(range(Config.EPOCHS_2), desc="Training Progress (minimize Euler residual)"):

        expected_reward_value = expected_lifetime_reward(policy_net = policy_net_Euler)
        reward_value = lifetime_reward(policy_net = policy_net_Euler)
        Euler_residual_value = Euler_residual(policy_net = policy_net_Euler)
        loss_value = training_step_Euler()

        expected_rewards.append(expected_reward_value)
        rewards.append(reward_value)
        losses.append(loss_value)
        Euler_residuals.append(Euler_residual_value)

    if (show_plot == True):
        print("Showing plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.plot(rewards, label="Lifetime Reward")
        plt.plot(expected_rewards, label="Expected Lifetime Reward")
        plt.title("Reward over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rewawrd")
        plt.grid()
        plt.show()
    return losses, Euler_residuals, expected_rewards, rewards


def train_Bellman(show_plot = False):
    """Trian a policy network and a value network to minimize the Euler-residual and Bellmen-residual at the same time

    Args:
        show_plot (bool, optional): if show_plot = True, it will show a plot after training process. Defaults to False.

    Returns:
        losses_policy, losses_value, Euler_residuals, expected_rewards, rewards
    """
    policy_net_Bellman = models.PolicyNet()
    value_net_Bellman = models.ValueNet()
    
    test_policy_expected(policy_net_Bellman, test_rounds=1)
    test_policy_expected(value_net_Bellman, test_rounds=1)
    optimizer_policy = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE_3_POLICY)
    optimizer_value = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE_3_VALUE)

    @tf.function
    def train_policy_net():
        with tf.GradientTape() as tape:
            loss = Euler_residual(policy_net = policy_net_Bellman)
        grads = tape.gradient(loss, policy_net_Bellman.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, policy_net_Bellman.trainable_variables) if g is not None]
        optimizer_policy.apply_gradients(grads_vars)
        return loss

    @tf.function
    def train_value_net():
        with tf.GradientTape() as tape:
            loss = Bellmen_residual(policy_net = policy_net_Bellman, value_net = value_net_Bellman)
        grads = tape.gradient(loss, value_net_Bellman.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, value_net_Bellman.trainable_variables) if g is not None]
        optimizer_value.apply_gradients(grads_vars)
        return loss


    # List to store loss and reward during training progress
    losses_policy = []
    losses_value = []
    expected_rewards = []
    rewards = []
    Euler_residuals = []

    for epoch in tqdm(range(Config.EPOCHS_3), desc="Training Progress (minimize Bellmen residual)"):
        
        expected_reward_value = expected_lifetime_reward(policy_net = policy_net_Bellman)
        reward_value = lifetime_reward(policy_net = policy_net_Bellman)
        Euler_residual_value = Euler_residual(policy_net = policy_net_Bellman)

        expected_rewards.append(expected_reward_value)
        rewards.append(reward_value)

        Euler_residuals.append(Euler_residual_value)
        
        loss_policy = train_policy_net()
        loss_value = train_value_net()

        losses_policy.append(loss_policy)
        losses_value.append(loss_value)


    if (show_plot == True):
        print("Showing plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.plot(rewards, label="Lifetime Reward(minimize Bellmen residual)")
        plt.plot(expected_rewards, label="Expected Lifetime Reward(minimize Bellmen residual)")
        plt.title("reward over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rewawrd")
        plt.grid()
        plt.show()
    return losses_policy, losses_value, Euler_residuals, expected_rewards, rewards


#------------------------------------------------ risky model extension-----------------------------------------------------

def train_lifetime_risky(show_plot = False):
    policy_net_liftime = models.PolicyNet_Risky()
    print(risky_lifetime_reward(policy_net = policy_net_liftime))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    @tf.function
    def training_step():
        with tf.GradientTape() as tape:
            loss = -risky_lifetime_reward(policy_net = policy_net_liftime, T = Config.T) # type: ignore

        grads = tape.gradient(loss, policy_net_liftime.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net_liftime.trainable_variables))
        return loss

    # List to store loss and reward during training progress
    losses = []
    expected_rewards = []
    rewards = []
    Euler_residuals = []

    for epoch in tqdm(range(Config.EPOCHS_1), desc="Training Progress (maximize lifetime reward)"):
        # expected_reward_value = risky_expected_lifetime_reward(policy_net = policy_net_liftime)
        reward_value = risky_lifetime_reward(policy_net = policy_net_liftime)
        Euler_residual_value = risky_Euler_residual(policy_net = policy_net_liftime)
        # Euler_residual_value = 0
        loss_value = training_step()

        # expected_rewards.append(expected_reward_value)
        rewards.append(reward_value)
        losses.append(loss_value)
        Euler_residuals.append(Euler_residual_value)

    if (show_plot == True):
        print("Showing plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.plot(rewards, label="Lifetime Reward")
        # plt.plot(expected_rewards, label="Expected Lifetime Reward")
        plt.title("Rewards over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Reward")
        plt.grid()
        plt.show()

    return losses, Euler_residuals, expected_rewards, rewards

def train_Euler_risky(show_plot = False):

    policy_net_Euler = models.PolicyNet_Risky()
    risky_expected_lifetime_reward(policy_net_Euler)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

    @tf.function
    def training_step_Euler():
        with tf.GradientTape() as tape:
            loss = risky_Euler_residual(policy_net = policy_net_Euler) 

        grads = tape.gradient(loss, policy_net_Euler.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net_Euler.trainable_variables))
        return loss

    # List to store loss and reward during training progress
    losses = []
    expected_rewards = []
    rewards = []
    Euler_residuals = []

    for epoch in tqdm(range(6000), desc="Training Progress (minimize Euler residual)"):

        expected_reward_value = risky_expected_lifetime_reward(policy_net = policy_net_Euler)
        reward_value = risky_lifetime_reward(policy_net = policy_net_Euler)
        Euler_residual_value = risky_Euler_residual(policy_net = policy_net_Euler)
        loss_value = training_step_Euler()

        expected_rewards.append(expected_reward_value)
        rewards.append(reward_value)
        losses.append(loss_value)
        Euler_residuals.append(Euler_residual_value)

    if (show_plot == True):
        print("Showing plot...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,8))
        plt.plot(rewards, label="Lifetime Reward")
        plt.plot(expected_rewards, label="Expected Lifetime Reward")
        plt.title("Reward over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Rewawrd")
        plt.grid()
        plt.show()
    return losses, Euler_residuals, expected_rewards, rewards