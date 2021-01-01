"""
Copyright (c) 2020, Shashwat Saxena.

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
from lobpy.datareader.orderbook import *
from pandas.tseries.holiday import AbstractHolidayCalendar, DateOffset, EasterMonday, GoodFriday, Holiday, MO, \
    nearest_workday, next_monday, next_monday_or_tuesday, USMartinLutherKingJr, USLaborDay, USThanksgivingDay, \
    USMemorialDay, USPresidentsDay
import datetime as dt


class XAssetAgent():
    """
    OBReader object specified for using LOBSTER files
    ----------
    params:
            ticker_str,
            date_str,
            time_start_str,
            time_end_str,
            num_levels_str,
            time_start_calc_str,
            time_end_calc_str


    Example usage:
    to create an object
    >>> agent = XAssetAgent("SYMBOL", "2012-06-21", "34200000", "57600000", "10")
    to set traded assets
    >>> # tbd
    to get all the traded readers
    >>> # tbd
    to add a new traded asset
    >>> # tbd
    """

    def __init__(self,
                 num_assets: int,
                 readers: list,
                 max_inventory: list):
        if not all([len(arg) == num_assets for arg in [readers, max_inventory]]):
            raise DataRequestError("Make sure all input lists match the number of assets")
        self.num_assets = num_assets
        self.readers = readers
        self.max_inventory = max_inventory
        self.current_inventory = [0 for _ in range(num_assets)]

    def inventory_ratio(self):
        return [curr_inv / max_inv for curr_inv, max_inv in zip(self.current_inventory, self.max_inventory)]


