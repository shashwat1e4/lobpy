"""
Copyright (c) 2021, Shashwat Saxena.

This module provides the helper functions and the class XAssetAgent, a reinforcement learning agent which will make
trading decisions based on LOBSTERReader data.
"""

######
# Imports
######
from typing import List

import torch.nn as nn
import torch.nn.functional as F

from lobpy.datareader.lobster import *
from scipy.interpolate import interp1d


class AgentState:
    """
    State object specified for categorizing limit order book orders
    ----------
    params:
            limit order book dataframe,
            agent quantity,
            agent cash


    Example usage:
    to create a state with agent cash and inventory
    >>> agent = AgentState(pd.DataFrame([1000, 1010]), 1, 500)
    to create a state from the reader itself
    >>> agents = AgentState(reader, 1, 500)
    """

    def __init__(self,
                 mid_price: float,
                 agent_position: float,
                 agent_cash: float,
                 time_to_eod: float,
                 qty_high_bound: float,
                 qty_low_bound: float,
                 sig_fig: float = 0.0001,
                 lo_intensity_bid: float = 1,
                 lo_intensity_ask: float = 1,
                 bid_fill_prob: float = 100,  # Check whether we can evaluate pdf/cdf directly
                 ask_fill_prob: float = 100,  # Check whether we can evaluate pdf/cdf directly
                 max_running_penalty: float = 0.01,  # Check this to make sure it kinda makes sense
                 liquidation_penalty: float = 100,  # Check this to make sure it kinda makes sense, but needs to be high
                 accumulated_penalty: float = 0.0):
        self.mid_price = mid_price  # measurable
        self.agent_position = agent_position  # agent-driven
        self.agent_cash = agent_cash  # agent-driven
        # Use this in the actual value function, rather than the estimate
        self.accumulated_penalty = accumulated_penalty  # agent-driven
        self.qty_high_bound = qty_high_bound  # agent-driven
        self.qty_low_bound = qty_low_bound  # agent-driven
        self.sig_fig = sig_fig  # agent-driven
        self.lo_intensity_bid = lo_intensity_bid  # secondary measure from LOB
        self.lo_intensity_ask = lo_intensity_ask  # secondary measure from LOB
        self.bid_fill_prob = bid_fill_prob  # agent-driven
        self.ask_fill_prob = ask_fill_prob  # agent-driven
        self.max_running_penalty = max_running_penalty
        self.liquidation_penalty = liquidation_penalty
        self.time_to_eod = time_to_eod

    @property
    def quantity_space(self):
        return np.linspace(self.qty_low_bound, self.qty_high_bound, num=int(1 / self.sig_fig))

    @property
    def est_log_penalized_value(self, is_bid_side=True) -> np.ndarray:
        # w(t, q) = exp(A) * z
        fill_prob = self.bid_fill_prob if is_bid_side else self.ask_fill_prob
        qty = self.quantity_space
        z = np.fromiter((np.exp(-self.max_running_penalty * fill_prob * quantity ^ 2) for quantity in qty), dtype=float)
        exp_a = np.zeros((len(qty), len(qty)))

        for i in range(len(qty)):
            for j in range(len(qty)):
                exp_a[i, j] = np.exp(- self.max_running_penalty * fill_prob * qty[i] if i == j else
                                     self.lo_intensity_bid / np.exp(1) if i + 1 == j else
                                     self.lo_intensity_ask / np.exp(1) if i - 1 == j else
                                     0)

        return exp_a @ z

    def est_running_penalty(self, bump: float = np.random.rand()) -> float:
        eta, zeta = 0.5, 0.5

        # h+/-(t, q) = w+/-(t, q) * exp(T - t) / k+/-
        h_bid = self.est_log_penalized_value(is_bid_side=True) * np.exp(self.time_to_eod) / self.bid_fill_prob
        h_ask = self.est_log_penalized_value(is_bid_side=False) * np.exp(self.time_to_eod) / self.ask_fill_prob

        return (eta * h_bid + zeta * h_ask) * (1 + bump)

    def est_value_fn(self) -> float:
        eod_penalty = 0 if self.time_to_eod > 0 else self.liquidation_penalty * self.agent_position ** 2

        running_penalty = interp1d(self.quantity_space, self.est_running_penalty(), kind='cubic')
        return (self.agent_cash + self.agent_position * self.mid_price
                - self.max_running_penalty * running_penalty(self.agent_position) - eod_penalty)


class PenaltyNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(PenaltyNetwork, self).__init__()
        self.layer_in = nn.Linear(input_size, 32)
        self.layer_1 = nn.Linear(32, 32)
        self.layer_out = nn.Linear(32, output_size)

    def forward(self, x):
        out_1 = F.relu(self.layer_in(x))
        out_2 = F.relu(self.layer_1(out_1))
        out_3 = self.layer_out(out_2)
        return out_3


class ValueNetEstimator:

    def __init__(self,
                 readers: List[LOBSTERReader],
                 qty_low_bound: int,
                 qty_high_bound: int,
                 max_cash: float,
                 neural_net: nn.Module,
                 loss_fn=nn.MSELoss,
                 sig_fig: float = 0.001):
        self.qty_low_bound = qty_low_bound
        self.qty_high_bound = qty_high_bound
        self.max_cash = max_cash
        self.readers = readers
        self.network = neural_net
        self.loss_fn = loss_fn
        self.sig_fig = sig_fig

    @staticmethod
    def _targets(training_input: np.ndarray) -> np.ndarray:
        return np.fromiter((state.est_value_fn for state in training_input), float)

    @staticmethod
    def _parse_states(readers: List[LOBSTERReader], qty_low_bound: int, qty_high_bound: int, max_cash: float) -> List[
        AgentState]:
        assert len(set([reader.ticker_str for reader in readers])) == 1

        states = list()
        for reader in readers:
            # Get all mid prices
            eod_time = reader.time_end - reader.time_start
            dataset = reader.mid_prices_over_time(reader.num_levels, 0, eod_time)
            qtys = np.linspace(qty_low_bound, qty_high_bound + 1, 1)
            cash = np.linspace(0, max_cash, 1)
            fill_prob = np.linspace(1, 100, 1)
            max_running_penalty = np.linspace(1, 10, 1)

            # Investigate use of Cython for this
            for qty, cash, fill, max_penalty in zip(qtys, cash, fill_prob, max_running_penalty):
                # We can train up the agent so that k+/- are the same. Exploration of RL agent can help with unequal
                # scenarios, given the lack of an explicit model. We also need to estimate low and high intensity from
                # the environment.
                states += [AgentState(price, qty, cash, time, qty_high_bound, qty_low_bound, lo_intensity_ask=buy_mo,
                                      lo_intensity_bid=sell_mo, bid_fill_prob=fill, ask_fill_prob=fill,
                                      max_running_penalty=10 ** -max_penalty)
                           for price, time, buy_mo, sell_mo in zip(dataset['Mid Prices'], dataset['Time'],
                                                                   dataset['Buy Intensity'], dataset['Sell Intensity'])]
        return states

    def train(self, steps: int):
        training_input = self._parse_states(self.readers, self.qty_low_bound, self.qty_high_bound, self.max_cash)
        targets = self._targets(training_input)
        for _ in range(steps):
            output = [self.network(train_in) for train_in in training_input]
            loss = self.loss_fn(list(targets), output)
            loss.backward()
            for param in self.network.parameters():
                param.data.add_(- param.grad / steps)
                param.grad.data.zero_()
