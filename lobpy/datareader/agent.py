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


def check(bool_expr: bool, error_str: str = 'Expression failed a Check clause!'):
    if bool_expr:
        return
    else:
        raise DataRequestError(error_str)


class Order:
    """
    Order object specified for categorizing limit order book orders
    ----------
    params:
            ticker_str,
            price,
            quantity,
            direction (long/short),


    Example usage:
    to create a long order object
    >>> agent = Order("AAPL", 10223535.345, 4, 1)
    to create a short order object
    >>> agent = Order("AAPL", 10223535.345, 4, -1)
    to create a long order object without specifying the direction
    """

    def __init__(self, ticker_str: str, price: float, abs_quantity: float, direction: int):
        self.ticker_str = ticker_str
        self.price = price
        if direction != 0 and abs_quantity != 0:
            self.direction = direction / abs(direction)
            self.quantity = abs(abs_quantity)
            return
        raise DataRequestError("Please enter a nonzero direction and/or quantity that corresponds to either a "
                               "positive or a negative number")

    def __init__(self, ticker_str: str, price: float, quantity: float):
        self.ticker_str = ticker_str
        self.price = price
        if quantity != 0:
            self.quantity = quantity
            self.direction = quantity / abs(quantity)
            return
        raise DataRequestError("Please enter a nonzero direction and/or quantity that corresponds to either a "
                               "positive or a negative number")


class XAssetAgent:
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
    >>> agent = XAssetAgent(1, [LOBSTERReader()], [10])
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
                 max_inventory: list,
                 cash_held: float):
        if not all([len(arg) == num_assets for arg in [readers, max_inventory]]):
            raise DataRequestError("Make sure all input lists match the number of assets")
        self.num_assets = num_assets
        self.readers = readers
        self.max_inventory = max_inventory
        self.current_inventory = [0 for _ in range(num_assets)]
        self.bid_orders = []
        self.ask_orders = []
        self.cash_held = cash_held

    def inventory_ratio(self):
        return [curr_inv / max_inv for curr_inv, max_inv in zip(self.current_inventory, self.max_inventory)]

    def distance_to_touch(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        dataset_period = [reader.mid_prices_over_time(num_levels_calc, time_offset_ms, time_period_ms) for reader in
                          self.readers]

        check(len(dataset_period.index) == 1, 'More than one record received!')
        mid_price = dataset_period["Mid Prices"].iloc[0]

        order: Order
        return (min([mid_price - order.price for order in enumerate(self.bid_orders)]) / mid_price - 1,
                min([order.price - mid_price for order in enumerate(self.ask_orders)]) / mid_price - 1)
