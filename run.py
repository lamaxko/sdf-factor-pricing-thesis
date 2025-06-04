import os

from pandas import read_clipboard
from apis.fred import FredApi
from models.lstm import MacroStateLSTM
from prep.factors import FactorCalc
from prep.tensors import TensorPreprocessor
from models.gan import SDFGan
from helpers.save import save_gan

def obtain_inputs(syear, eyear, recalc=False):

    macro_stationary = FredApi().data_stationary.copy()

    state_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t = state_rnn.extract_from_dataframe(macro_stationary)

    moment_rnn = MacroStateLSTM(num_states=4, hidden_size=32, seq_len=12)
    h_t_g = moment_rnn.extract_from_dataframe(macro_stationary)

    I_ti = FactorCalc(syear=syear, eyear=eyear, lag=1, recalc=recalc).panel.copy()

    # x_omega, x_g, y
    input = TensorPreprocessor(h_t, h_t_g, I_ti).preprocess()
    return input

def run_gan(input):

    x_omega = input.x_omega
    x_g = input.x_g
    y = input.y

    gan = SDFGan(input_dim_omega=x_omega.shape[1], input_dim_g=x_g.shape[1])
    gan.fit(x_omega, x_g, y, n_epochs=100, inner_steps=5)

    omega_final, g_final = gan.extract_outputs(x_omega, x_g)

    panel = input.panel.copy()
    panel['omega'], panel['g'] = omega_final, g_final

    training_log = gan.get_training_log()
    return panel, training_log

def oos_backtest(test_syear=2006, test_eyear=2023, save_dir=r'out/', recalc=False):

    syear = 2000
    for eyear in range(test_syear, test_eyear + 1):
        inputs = obtain_inputs(syear, eyear)
        panel, training_log = run_gan(inputs)
        save_gan(panel, training_log, save_dir, syear, eyear)
    return

def is_backtest(syear=2000, eyear=2023, save_dir=r'out/', recalc=False):
    inputs = obtain_inputs(syear, eyear)
    panel, training_log = run_gan(inputs)
    save_gan(panel, training_log, save_dir, syear, eyear)

if __name__ == "__main__":
    # is_backtest(recalc=True)
    oos_backtest(test_eyear=2024)
    
