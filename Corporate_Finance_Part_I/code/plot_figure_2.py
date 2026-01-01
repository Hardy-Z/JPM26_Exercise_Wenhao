# Produce Plot 2 in the report
from config import *
Config.K_INITIAL = 100.0

from utils import *
import models
from tqdm import tqdm
import utils_train

plt.rcParams['font.size'] = 25
def main():
    losses_1, Euler_residuals_1, expected_rewards_1, rewards_1 = utils_train.train_lifetime_risky()
    print("(life-time reward training) Euler-residual at last step is: ", Euler_residuals_1[-1].numpy())
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    losses_2, Euler_residuals_2, expected_rewards_2, rewards_2 = utils_train.train_Euler_risky()
    print("(Euler--residual training) Euler-residual at last step is: ", Euler_residuals_2[-1].numpy())

    plt.figure(figsize=(12,8))
    plt.plot(rewards_1, label="Lifetime Reward (lifetime trained policy)", linewidth=0.9, alpha=0.7, color = 'lightcoral')
    plt.plot(expected_rewards_1, label="Expected Lifetime Reward (lifetime trained policy)",linewidth=1.1, color = 'red')
    plt.plot(rewards_2, label="Lifetime Reward (Euler trained policy)",linewidth=0.9, alpha=0.7, color = 'lightblue')
    plt.plot(expected_rewards_2, label="Expected Lifetime Reward (Euler trained policy)",linewidth=1.1, color = 'blue')
    plt.title("Lifetime Reward over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend(loc ="lower right", fontsize=18)
    plt.grid()
    plt.savefig("Figure 2 (a).png")

    plt.figure(figsize=(12,8))
    plt.plot(Euler_residuals_1, label="Euler Residuals (lifetime trained policy)", linewidth=1.1, color = 'red')
    plt.plot(Euler_residuals_2, label="Euler Residuals (Euler trained policy)", linewidth=1.1, color = 'blue')
    plt.title("Euler Residuals over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Euler Residuals")
    plt.legend(loc ="upper right")
    plt.grid()
    plt.savefig("Figure 2 (b).png")

if __name__ == "__main__":
    main()
