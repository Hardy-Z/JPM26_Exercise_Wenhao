class Config:
    ## random seed
    RANDOM_SEED = 0
    TF_RANDOM_SEED = 0

    # hyperparameters in DNN
    HIDDEN_UNITS = 128
    HIDDEN_UNITS_RISKY = 256
    LEARNING_RATE_1 = 4e-4
    LEARNING_RATE_2 = 9e-4
    LEARNING_RATE_3_POLICY = 3e-4
    LEARNING_RATE_3_VALUE = 2e-4
    EPOCHS_1 = 6000
    EPOCHS_2 = 6000
    EPOCHS_3 = 6000
    # model parameters
    THETA = 0.9   # capital productivity exponent in profit (if Cobb-Douglas)
    PHI = 0.1     # adjustment cost scale
    DELTA = 0.20   # depreciation rate
    RHO = 0.7      # AR(1) persistence of shock.
    TAU = 0.1
    SIGMA_EPS = 0.001  # std dev of shock innovation
    Z_HIGH = 4.0  # upper bound of z
    Z_LOW = -4   # lower bound of z

    BETA = 0.8  # discount factor
    K_INITIAL = 10.0
    # K_INITIAL = 100.0
    Z_INITIAL = 2.0
    B_INITIAL = -500

    T = 20  # simulation length