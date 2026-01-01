# Produce Plot 1 in the report
import matplotlib.pyplot as plt
import utils_train
import tensorflow as tf
import os
plt.rcParams['font.size'] = 25
def main():

    losses_1, Euler_residuals_1, expected_rewards_1, rewards_1 = utils_train.train_lifetime()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    losses_2, Euler_residuals_2, expected_rewards_2, rewards_2 = utils_train.train_Euler()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    losses_3_policy, losses_3_value, Euler_residuals_3, expected_rewards_3, rewards_3 = utils_train.train_Bellman()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


    plt.figure(figsize=(12,8))
    plt.plot(rewards_1, label="Lifetime Reward (lifetime trained policy)", linewidth=0.9, alpha=0.7, color = 'lightcoral')
    plt.plot(expected_rewards_1, label="Expected Lifetime Reward (lifetime trained policy)",linewidth=1.1, color = 'red')
    plt.plot(rewards_2, label="Lifetime Reward (Euler trained policy)",linewidth=0.9, alpha=0.7, color = 'lightblue')
    plt.plot(expected_rewards_2, label="Expected Lifetime Reward (Euler trained policy)",linewidth=1.1, color = 'blue')
    plt.plot(rewards_3, label="Lifetime Reward (Bellman trained policy)",linewidth=0.9, alpha=0.7, color = 'wheat')
    plt.plot(expected_rewards_3, label="Expected Lifetime Reward (Bellman trained policy)",linewidth=1.1, color = 'orange')
    plt.title("Lifetime Reward over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend(loc ="lower right", fontsize=18)
    plt.grid()
    plt.savefig("Figure 1 (a).png")

    plt.figure(figsize=(12,8))
    plt.plot(Euler_residuals_1, label="Euler Residuals (lifetime trained policy)", linewidth=1.1, color = 'red')
    plt.plot(Euler_residuals_2, label="Euler Residuals (Euler trained policy)", linewidth=1.1, color = 'blue')
    plt.plot(Euler_residuals_3, label="Euler Residuals (Bellman trained policy)", linewidth=1.1, color = 'orange')
    plt.title("Euler Residuals over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Euler Residuals")
    plt.legend(loc ="upper right")
    plt.grid()
    plt.savefig("Figure 1 (b).png")

    plt.figure(figsize=(12,8))
    plt.plot(losses_3_value, label="Bellmen Residuals for value function", linewidth=1.1, color = 'orange')
    plt.title("Bellmen Residuals over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Bellmen Residuals")
    plt.legend()
    plt.grid()
    plt.savefig("Figure 1 (c).png")


if __name__ == "__main__":
    main()