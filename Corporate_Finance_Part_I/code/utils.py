import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from config import *


tfd = tfp.distributions

def pi(k, z):
    """
    Profit function:  pi(k,z) = z ^theta
    """
    return z * tf.where(k>0, tf.pow(k, Config.THETA), 0.0)


def psi(I, k):
    """
    Investment adjustment cost function: psi(I, k)= phi/2k
    """
    return 0.5 * Config.PHI * tf.square(I)/ tf.where(k > 0.0, k + 1e-7, k- 1e-7)  # handle k=0 case

def e(k,I,z):
    """
    Net value at time 
    """
    return pi(k,z) - psi(I,k) - I


def next_z(z, rho = Config.RHO):
    """
    ln(z_{t+1}) = rho * ln(z_t) + eps
    """
    z = tf.convert_to_tensor(z, dtype=tf.float32)

    dist = tfd.TruncatedNormal(
        loc=0.0,
        scale=Config.SIGMA_EPS,
        low=Config.Z_LOW,
        high=Config.Z_HIGH
    )
    eps = dist.sample(tf.shape(z))
    log_z = tf.math.log(z)
    log_z_next = rho * log_z + eps
    z_next = tf.math.exp(log_z_next)
    return z_next


def next_k(k,I):
    """ k_{t+1} = k_t(1-delta) + I_t
    """
    return (1-Config.DELTA)*k + I





def expected_z_next(Ez_t, rho = Config.RHO, sigma_eps = Config.SIGMA_EPS):
    """ Since e^Z follows AR(1), so E[z_{t+1} | z_t] = exp{eps/2}z_t^rho
    """
    return tf.pow(Ez_t, rho) * tf.exp(0.5 * sigma_eps**2)

@tf.function
def expected_lifetime_reward(policy_net, k_0=Config.K_INITIAL, z_0=Config.Z_INITIAL, T=Config.T, beta=Config.BETA):
    """calculate expected life time reward given a policy network

    Returns:
        _type_: float32 scalar tensor
    """
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)
    R = tf.constant(0.0, dtype=tf.float32)

    for t in tf.range(T):
        I = I_pred(k, z, policy_net = policy_net)


        reward_t = e(k, I, z)
        R += tf.pow(Config.BETA, tf.cast(t, tf.float32)) * reward_t

        k = next_k(k, I)
        z = expected_z_next(z)
        tf.debugging.assert_all_finite(z, "z has NaN/Inf")
        tf.debugging.assert_all_finite(I, "I has NaN/Inf")
        tf.debugging.assert_all_finite(k, "k has NaN/Inf")
    return R

@tf.function
def lifetime_reward(k_0=Config.K_INITIAL, z_0=Config.Z_INITIAL, T=Config.T, beta=Config.BETA, policy_net = None):
    """calculate  life time reward given a policy network

    Returns:
        _type_: float32 scalar tensor
    """
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)

    R = tf.constant(0.0, dtype=tf.float32)

    for t in tf.range(T):

        I = I_pred(k, z, policy_net = policy_net)
        reward_t = e(k, I, z)
        R += tf.pow(beta, tf.cast(t, tf.float32)) * reward_t
        k = next_k(k, I)
        z = next_z(z)

        tf.debugging.assert_all_finite(z, "z has NaN/Inf")
        tf.debugging.assert_all_finite(I, "I has NaN/Inf")
        tf.debugging.assert_all_finite(k, "k has NaN/Inf")

    return R



def e_I(k, I, z):
    """Partial derivative of e(k,I,z) w.r.t. I
    """
    k = tf.cast(k, tf.float32)
    I = tf.cast(I, tf.float32)
    # psi_I = d psi / d I
    psi_I = Config.PHI * I / (k + 1e-8)
    return -psi_I - 1.0


def e_k(k, I, z):
    """Partial derivative of e(k,I,z) w.r.t. k
    """
    k = tf.cast(k, tf.float32)
    I = tf.cast(I, tf.float32)
    z = tf.cast(z, tf.float32)
    # pi_k = d pi / d k
    pi_k = Config.THETA * z * tf.where(k>0, tf.pow(k, Config.THETA-1.0), 0.0)
    # psi_k = d psi / d k
    psi_k = -0.5 * Config.PHI * tf.square(I) / (tf.square(k)+ 1e-8)
    return pi_k - psi_k

def I_pred(k, z, policy_net, if_risky = False, b = None):
    """
    given scalar k, z, use policy_net to predict I (scalar output)
    """
    k = tf.cast(k, tf.float32)
    z = tf.cast(z, tf.float32)
    if (if_risky == False):
        state = tf.stack([k, z], axis=-1)      # (2,)
        state = tf.expand_dims(state, axis=0)  # (1, 2)
        I_pred = policy_net(state)             # (1, 1)
        I_pred = tf.squeeze(I_pred)   
    else:
        tf.debugging.assert_all_finite(b, "b contains NaN or Inf")
        b = tf.cast(b, tf.float32)
        state = tf.stack([k, z, b], axis=-1)      # (3,)
        state = tf.expand_dims(state, axis=0)  # (1, 3)
        I_pred = policy_net(state)             # (1, 1)
    

    return I_pred



@tf.function
def value_pred(k, z, value_net):
    """given scalar k, z, use value_net to predict V(k,z) (scalar output)
    """
    k = tf.cast(k, tf.float32)
    z = tf.cast(z, tf.float32)
    state = tf.stack([k, z], axis=-1)      
    state = tf.expand_dims(state, axis=0)  
    V_pred = value_net(state)            
    V_pred = tf.squeeze(V_pred)        

    return V_pred


@tf.function
def Euler_residual(k_0 = Config.K_INITIAL, z_0 = Config.Z_INITIAL, T = Config.T, policy_net = None, value_net = None):
    """Calculate Euler-residual given a policy net
    Returns:
        tf.float: loss value
    """
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)
    loss = tf.constant(0.0, dtype=tf.float32)
    for t in range(T):

        I = I_pred(k, z, policy_net = policy_net)
        

        k_next = next_k(k, I)
        z_next = expected_z_next(z)

        I_next = I_pred(k_next, z_next, policy_net = policy_net)
        term_3 = e_I(k, I, z) + Config.BETA * (e_k(k_next, I_next, z_next) - (1 - Config.DELTA)*e_I(k_next, I_next, z_next))
        loss += tf.square(term_3)

        k = k_next
        z = z_next
    return loss


def test_policy_expected(policy_net, test_rounds=1):
    rewards = []
    for _ in range(test_rounds):

        k = tf.constant(Config.K_INITIAL, dtype=tf.float32)
        z = tf.constant(Config.Z_INITIAL, dtype=tf.float32)
        R = tf.constant(0.0, dtype=tf.float32)

        for t in tf.range(Config.T):

            I = I_pred(k, z, policy_net = policy_net)

            reward_t = e(k, I, z)

            R += tf.pow(Config.BETA, tf.cast(t, tf.float32)) * reward_t

            k = next_k(k, I)
            z = next_z(z)
            tf.debugging.assert_all_finite(z, "z has NaN/Inf")
            tf.debugging.assert_all_finite(I, "I has NaN/Inf")
            tf.debugging.assert_all_finite(k, "k has NaN/Inf")
        return R


def Bellmen_residual(policy_net, value_net, k_0=Config.K_INITIAL, z_0= Config.Z_INITIAL):
    """Calculate Bellmen-residual given a policy net and a value net


    Returns:
        tf.float: loss value
    """
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)

    loss = tf.constant(0.0, dtype=tf.float32)

    for t in range(Config.T):
        I = I_pred(k, z, policy_net = policy_net)

        k_next = next_k(k, I)
        Ez_next = expected_z_next(z, Config.RHO, Config.SIGMA_EPS)
        z_next = next_z(z)

        V_hat = value_pred(k = k, z = z, value_net = value_net)
        V_hat_next= value_pred(k = k_next, z = Ez_next, value_net = value_net)

        term_Bellmen = V_hat - (e(k, I, z) + Config.BETA * V_hat_next)
        loss += tf.square(term_Bellmen)

        k = k_next
        z = Ez_next

        tf.debugging.assert_all_finite(z, "z has NaN/Inf")
        tf.debugging.assert_all_finite(I, "I has NaN/Inf")
        tf.debugging.assert_all_finite(k, "k has NaN/Inf")
        tf.debugging.assert_all_finite(loss, "loss has NaN/Inf")
    return loss



# -------------------------------------functions for risky model-------------------------------------#


def next_b(b,D):
    return b + D

def risky_debt_rate(z, k_next, b_next):
    """calculate risky intrest rate
    """
    return tf.nn.relu( -1*z - 0.5*k_next + 1*b_next) + 0.01  # ensure rate is positive

def risky_e(k,b,z,I,D):
    """ net value with risky debt
    """
    b_next = next_b(b,D)
    k_next = next_k(k,I)
    r_tilde = risky_debt_rate(z, k_next, b_next)
    term_1 =  (1-Config.TAU)*pi(k,z) - psi(I,k) - I
    term_2 = b_next/(1 + r_tilde)
    term_3 = Config.TAU*(r_tilde) *b_next*Config.BETA/(1+r_tilde) - b
    return term_1 + term_2 + term_3



@tf.function
def risky_expected_lifetime_reward(policy_net, k_0=Config.K_INITIAL, z_0=Config.Z_INITIAL, b_0=Config.B_INITIAL, T=Config.T, beta=Config.BETA):
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)
    R = tf.constant(0.0, dtype=tf.float32)
    b = tf.constant(b_0, dtype=tf.float32)

    for t in tf.range(T):
        u = I_pred(k, z, policy_net = policy_net, if_risky = True, b = b)
        I = u[0,0]
        D = u[0,1]
        reward_t = risky_e(k = k,b = b,z=z,I=I,D=D)
        R += tf.pow(beta, tf.cast(t, tf.float32)) * reward_t
        k = next_k(k, I)
        z = expected_z_next(z)
        b = next_b(b, D=D)
        tf.debugging.assert_all_finite(z, "z has NaN/Inf")
        tf.debugging.assert_all_finite(I, "I has NaN/Inf")
        tf.debugging.assert_all_finite(k, "k has NaN/Inf")
        tf.debugging.assert_all_finite(b, "b has NaN/Inf")
        tf.debugging.assert_all_finite(reward_t, "reward_t has NaN/Inf")
    return R

@tf.function
def risky_lifetime_reward(policy_net, k_0=Config.K_INITIAL, z_0=Config.Z_INITIAL, b_0=Config.B_INITIAL, T=Config.T, beta=Config.BETA):
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)
    R = tf.constant(0.0, dtype=tf.float32)
    b = tf.constant(b_0, dtype=tf.float32)

    for t in tf.range(T):
        u = I_pred(k, z, policy_net = policy_net, if_risky = True, b = b)
        I = u[0,0]
        D = u[0,1]
        reward_t = risky_e(k = k,b = b,z=z,I=I,D=D)
        R += tf.pow(beta, tf.cast(t, tf.float32)) * reward_t
        k = next_k(k, I)
        z = next_z(z)
        b = next_b(b, D=D)
        tf.debugging.assert_all_finite(z, "z has NaN/Inf")
        tf.debugging.assert_all_finite(I, "I has NaN/Inf")
        tf.debugging.assert_all_finite(k, "k has NaN/Inf")
    return R


@tf.function
def risky_Euler_residual(policy_net, k_0 = Config.K_INITIAL, z_0 = Config.Z_INITIAL, b_0 = Config.B_INITIAL,T = Config.T, ):
    k = tf.constant(k_0, dtype=tf.float32)
    z = tf.constant(z_0, dtype=tf.float32)
    b = tf.constant(b_0, dtype=tf.float32)
    loss = tf.constant(0.0, dtype=tf.float32)
    for t in range(T):

        u = I_pred(k, z, policy_net = policy_net,if_risky=True, b = b)
        I = u[0,0]
        D = u[0,1]


        k_next = next_k(k, I)
        z_next = expected_z_next(z)
        b_next = next_b(b,D)

        u_next = I_pred(k_next, z_next, policy_net = policy_net, if_risky=True, b = b)
        I_next = u_next[0,0]
        D_next = u_next[0,1]

        term_1 = e_I(k, I, z) + Config.BETA * (risky_e_k(k_next, I_next, z_next, I_next, D_next) - (1 - Config.DELTA)*e_I(k_next, I_next, z_next))
        term_2 = Config.BETA*(risky_e_b(k_next,b_next,z_next,I_next,D_next) - risky_e_D(k_next,b_next,z_next,I_next,D_next)) + risky_e_D(k, b,z,I,D)

        loss +=  tf.square(term_1) + tf.square(term_2)

        k = k_next
        z = z_next
    return loss

def risky_e_k(k, b, z, I, D):
    """derivative of net value w.r.t. k
    """
    k = tf.cast(k, tf.float32)
    I = tf.cast(I, tf.float32)
    b = tf.cast(b, tf.float32)
    z = tf.cast(z, tf.float32)
    b_next = next_b(b, D)
    k_next = next_k(k, I)
    r_tilde = risky_debt_rate(z, k_next = k_next, b_next= b_next)
    # pi_k = d pi / d k
    pi_k = Config.THETA * z * tf.where(k>0, tf.pow(k, Config.THETA-1.0), 0.0)
    # psi_k = d psi / d k
    psi_k = -0.5 * Config.PHI * tf.square(I) / (tf.square(k)+ 1e-8)

    term_1 = (1-Config.TAU)*pi_k - psi_k
    term_2 = - (b+D)/(1+r_tilde)**2  * tf.where(r_tilde> 0.01, -0.5, 0.0) * (1-Config.DELTA)
    return term_1 + term_2


def risky_e_b(k, b, z, I, D):
    """derivative of net value w.r.t. b
    """
    k = tf.cast(k, tf.float32)
    I = tf.cast(I, tf.float32)
    b = tf.cast(b, tf.float32)
    z = tf.cast(z, tf.float32)

    b_next = next_b(b, D)
    k_next = next_k(k, I)

    r_tilde = risky_debt_rate(z, k_next, b_next)
    r_tilde_b = tf.where(r_tilde > 0.01, 1.0, 0.0)
    term_1 = ((1+r_tilde) - r_tilde_b*(b_next))/(1+r_tilde)**2
    term_2 = (Config.TAU* (r_tilde_b*b_next + r_tilde)*(1+r_tilde)*Config.BETA - (Config.TAU*r_tilde*b_next)*r_tilde_b*Config.BETA) /((1+r_tilde)*Config.BETA)**2
    term_3 = -1
    return term_1 + term_2 + term_3

def risky_e_D(k, b, z, I, D):
    """derivative of net value w.r.t. D
    """
    k = tf.cast(k, tf.float32)
    I = tf.cast(I, tf.float32)
    b = tf.cast(b, tf.float32)
    z = tf.cast(z, tf.float32)
    b_next = next_b(b, D)
    k_next = next_k(k, I)
    r_tilde = risky_debt_rate(z, k_next, b_next)

    r_tilde_D = tf.where(r_tilde> 0.01, 1.0, 0.0)

    term_1 = (1+r_tilde - b_next*r_tilde_D)/(1+r_tilde)**2
    term_2 = (Config.TAU* (r_tilde_D*b_next + r_tilde)*(1+r_tilde)*Config.BETA - (Config.TAU*r_tilde*b_next)*r_tilde_D*Config.BETA) /((1+r_tilde)*Config.BETA)**2
    return term_1 + term_2
