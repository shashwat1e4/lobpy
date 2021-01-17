"""
Copyright (c) 2021, Shashwat Saxena.

This module provides the helper functions and the class XAssetAgent, a reinforcement learning agent which will make
trading decisions based on LOBSTERReader data.
"""

######
# Imports
######
import csv
import math
import warnings
from sortedcontainers import SortedDict
from typing import Callable

import numpy as np
import pandas as pd
from lobpy.datareader.lobster import *
from pandas.tseries.holiday import AbstractHolidayCalendar, DateOffset, EasterMonday, GoodFriday, Holiday, MO, \
    nearest_workday, next_monday, next_monday_or_tuesday, USMartinLutherKingJr, USLaborDay, USThanksgivingDay, \
    USMemorialDay, USPresidentsDay
import datetime as dt


class State:
    """
    State object specified for categorizing limit order book orders
    ----------
    params:
            limit order book dataframe,
            agent quantity,
            agent cash


    Example usage:
    to create a state with agent cash and inventory
    >>> agent = State(pd.DataFrame([1000, 1010]), 1, 500)
    to create a state from the reader itself
    >>> agents = State(reader, 1, 500)
    """

    def __init__(self,
                 mid_price: float,
                 agent_position: float,
                 agent_cash: float,
                 time_to_eod: float,
                 accumulated_penalty: float = 0.0):
        self.mid_price = mid_price
        self.agent_position = agent_position
        self.agent_cash = agent_cash
        self.accumulated_penalty = accumulated_penalty
        self.time_to_eod = time_to_eod


def value_fn(state: State, regularization_param: float = 1.0):
    return max(state.agent_cash + state.agent_cash * state.agent_position
               - regularization_param * state.accumulated_penalty, 0)
