"""
Copyright (c) 2018, University of Oxford, Rama Cont and ETH Zurich, Marvin S. Mueller

This module provides the helper functions and the class LOBSTERReader, a subclass of OBReader to read in limit order
book data in lobster format.
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


# LOBSTER specific file name functions
GDRIVE_COLAB_MOUNT = "/content/gdrive/My Drive/Colab Notebooks/"
PRICE_IN_USD = 10000


def _split_lobster_filename(filename):
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """
    filename2, _ = filename.split(".")
    ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels = filename2.split("_")
    return ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels


def split_lobster_filename(filename):
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """
    return _split_lobster_filename(filename)


def _split_lobster_filename_core(filename):
    """ splits the LOBSTER-type filename into Ticker, Date, Time Start, Time End, File Type, Number of Levels """
    filename2, _ = filename.split(".")
    ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels = filename2.split("_")
    return ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels


def _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels):
    return "_".join((ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels))


def create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels):
    return _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str, file_type_str, num_levels)


def _get_time_stamp_before(time_stamps, time_stamp):
    ''' Returns the value and index of the last time point in time_stamps before or equal time_stamp '''
    time = time_stamps[0]
    index = int(0)
    if time == time_stamp:
        # time_stamp found at index 0
        return time, index
    if time > time_stamp:
        raise LookupError("Time stamp data start at {} which is after time_stamps: {}".format(time, time_stamp))
    for ctr, time_now in enumerate(time_stamps[1:]):
        if time_now > time_stamp:
            return time, ctr
        time = time_now

    return time, ctr + 1


def _calc_mid_price(bid: float, ask: float):
    if np.isnan(bid) or np.abs(bid) > 99999:
        if np.isnan(ask) or np.abs(ask) > 99999:
            return np.nan
        else:
            return ask
    elif np.isnan(ask) or np.abs(ask) > 99999:
        return bid
    return (bid + ask) / 2.


def _calc_spread(bid: float, ask: float):
    if np.isnan(bid) or np.abs(bid) > 99999 or np.isnan(ask) or np.abs(ask) > 99999:
        return np.nan

    return ask - bid


def _calc_touch_volume(bid_volume: float, ask_volume: float):
    is_bid_nan = np.isnan(bid_volume)
    is_ask_nan = np.isnan(ask_volume)
    if is_bid_nan:
        if is_ask_nan:
            return np.nan
        else:
            return ask_volume
    else:
        if is_ask_nan:
            return bid_volume
        else:
            return (bid_volume + ask_volume) / 2.


def _prev_workday(day: str, cal: AbstractHolidayCalendar) -> pd.Timestamp:
    ts = pd.Timestamp(day)
    return ts + pd.offsets.CustomBusinessDay(-1, calendar=cal)


def _filter_time(lst, start_time, end_time):
    return [a for a in lst if start_time <= a < end_time]


def batch_data(dataset_period: pd.DataFrame, computation, interval_ms, time_offset_ms,
               time_period_ms):
    time_interval = time_period_ms - time_offset_ms
    slots = np.floor(time_interval / interval_ms)
    slotted_list = []
    data = []
    time_periods = dataset_period['Time']
    for i in range(slots):
        slotted_list.append(
            _filter_time(time_periods, time_offset_ms + interval_ms * i, time_offset_ms + interval_ms * (i + 1)))
    for slot in slots:
        selected_frames = pd.loc[(pd['Time'].isin(slot))]
        data.append(computation(selected_frames))
    # now bucket into times, and then find SD of each bucket!
    return data


def _calc_returns(selected_frames) -> list:
    mid_prices, shifted_prices = selected_frames['Mid_Prices'], selected_frames['Shifted Prices']
    return [np.log(a, b) for a, b in zip(mid_prices, shifted_prices)]


class HOLUSD(AbstractHolidayCalendar):
    """
    A representation of all the holidays in the US trading calendar.
    ----------
    params:
            year


    Example usage:
    to create an object
    >>> holidays = HOLUSD()

    """
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]


class HOLGBP(AbstractHolidayCalendar):
    """
        A representation of all the holidays in the US trading calendar.
        ----------
        params:
                year


        Example usage:
        to create an object
        >>> holidays = HOLGBP()

    """
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May Bank Holiday', month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Spring Bank Holiday', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer Bank Holiday', month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas Day', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26, observance=next_monday_or_tuesday)
    ]

class LOBSTERReader(OBReader):
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
    >>> lobreader = LOBSTERReader("SYMBOL", "2012-06-21", "34200000", "57600000", "10")
    read market depth on uniform time grid with num_observation number of observations
    >>> dt, time_stamps, depth_bid, depth_ask = lobreader.load_marketdepth(num_observations)
    read price process on that time grid specified above
    >>> dt2, time_stamps2, price_mid, price_bid, price_ask = lobreader.load_marketdepth(None)

    """

    def __init__(
            self,
            ticker_str,
            date_str,
            time_start_str,
            time_end_str,
            num_levels_str,
            time_start_calc_str="",
            time_end_calc_str="",
            num_levels_calc_str="",
            gdrive_mount=False,
    ):
        __prefix = GDRIVE_COLAB_MOUNT if gdrive_mount else str()
        self.ticker_str = ticker_str
        self.date_str = date_str
        self.lobfilename = __prefix + _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str,
                                                               ORDERBOOK_FILE_ID, num_levels_str)
        self.msgfilename = __prefix + _create_lobster_filename(ticker_str, date_str, time_start_str, time_end_str,
                                                               MESSAGE_FILE_ID, num_levels_str)
        self.time_start = int(time_start_str)
        self.time_end = int(time_end_str)
        self.num_levels = int(num_levels_str)
        self.time_start_calc = int(time_start_str)
        self.time_end_calc = int(time_end_str)
        self.num_levels_calc = int(num_levels_str)
        if not (num_levels_calc_str == ""):
            self.num_levels_calc = int(num_levels_calc_str)
        self.data = dict()
        if not (time_start_calc_str == ""):
            self.time_start_calc = int(time_start_calc_str)
        if not (time_end_calc_str == ""):
            self.time_end_calc = int(time_end_calc_str)

    # FIXME: Use Functool Wrap to correct the function reference on the print statement
    def _calc_occurrence_rate(self, order_type: int, num_levels_calc_str=str(), write_outputfile=False):
        print("Starting computation of average cancellation rate in file %s." % self.lobfilename)

        num_levels_calc = self.get_num_levels(num_levels_calc_str)

        with open(self.lobfilename + ".csv", newline='') as orderbookfile, open(
                ".".join((self.msgfilename, 'csv')), newline='') as messagefile:
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            num_lines = sum(1 for row in lobdata)
            print("Loaded successfully. Number of lines: " + str(num_lines))
            print("Start calculation.")
            orderbookfile.seek(0)
            messagefile.seek(0)
            total_time = self.time_end_calc - self.time_start_calc
            cancellation_count = SortedDict()
            cancellation_hist = SortedDict()
            # data are read as list of strings
            for rowLOB, rowMES in zip(lobdata, messagedata):
                if int(rowLOB[1]) == order_type:
                    cancellation_count.setdefault(float(rowMES[4]) - self._calc_midprice(rowLOB), []).append(1)

            for rel_price, entries in cancellation_count.items():
                cancellation_hist.setdefault(rel_price / PRICE_IN_USD, sum(entries) / total_time)

            relative_prices = np.array(cancellation_hist.keys())
            cancellation_rates = np.array(cancellation_hist.values())
            if write_outputfile:
                self._write_output_to_csv(num_levels_calc, relative_prices, cancellation_rates)
            return relative_prices, cancellation_rates

    def get_partial_cancellation_rates_tt(self, num_levels_calc_str=str(), write_outputfile=False):
        """ Computes the average cancellation rate over the course of the trading day against the distance to best
        quote. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean
        computation
        ----------
        args:
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (distance_from_best_quote, mean_cancellation_rate)  in format of numpy arrays
        """
        return self._calc_occurrence_rate(2, num_levels_calc_str=num_levels_calc_str, write_outputfile=write_outputfile)

    def get_total_cancellation_rates_tt(self, num_levels_calc_str=str(), write_outputfile=False):
        """ Computes the average cancellation rate over the course of the trading day against the distance to best
        quote. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean
        computation
        ----------
        args:
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (distance_from_best_quote, mean_cancellation_rate)  in format of numpy arrays
        """
        return self._calc_occurrence_rate(3, num_levels_calc_str=num_levels_calc_str, write_outputfile=write_outputfile)

    def get_all_cancellation_rates_tt(self, num_levels_calc_str=str(), write_outputfile=False):
        """ Computes the average cancellation rate over the course of the trading day against the distance to best
        quote. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean
        computation
        ----------
        args:
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (distance_from_best_quote, mean_cancellation_rate)  in format of numpy arrays
        """
        partial_prices, partial_rates = self._calc_occurrence_rate(2, num_levels_calc_str=num_levels_calc_str,
                                                                   write_outputfile=False)
        total_prices, total_rates = self._calc_occurrence_rate(2, num_levels_calc_str=num_levels_calc_str,
                                                               write_outputfile=False)

        combined_prices = np.append(partial_prices, [total_prices])
        combined_rates = np.append(partial_rates, [total_rates])

        if write_outputfile:
            self._write_output_to_csv(self.get_num_levels(num_levels_calc_str), combined_prices, combined_rates)
        return combined_prices, combined_rates

    def get_arrival_rates_tt(self, num_levels_calc_str=str(), write_outputfile=False):
        """ Computes the average cancellation rate over the course of the trading day against the distance to best
        quote. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean
        computation
        ----------
        args:
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (distance_from_best_quote, mean_cancellation_rate)  in format of numpy arrays
        """
        return self._calc_occurrence_rate(1, num_levels_calc_str=num_levels_calc_str, write_outputfile=write_outputfile)

    def _calc_midprice(self, orderbook_row, weighted=False):
        ask_price, ask_qty, bid_price, bid_qty = [float(x) for x in orderbook_row[0:4]]
        bid_price = ask_price if abs(bid_price) == 9999999999 else bid_price
        ask_price = bid_price if abs(ask_price) == 9999999999 else ask_price
        ask_qty, bid_qty = (ask_qty, bid_qty) if weighted else (1, 1)

        return (ask_price * ask_qty + bid_price * bid_qty) / (ask_qty + bid_qty)

    def set_timecalc(self, time_start_calc_str, time_end_calc_str):
        self.time_start_calc = int(time_start_calc_str)
        self.time_end_calc = int(time_end_calc_str)
        return True

    def create_filestr(self, identifier_str, num_levels=None):
        """ Creates lobster type file string """
        if num_levels is None:
            num_levels = self.num_levels
        return _create_lobster_filename(self.ticker_str, self.date_str, str(self.time_start_calc),
                                        str(self.time_end_calc), identifier_str, str(num_levels))

    def average_profile_tt(self, num_levels_calc_str="", write_outputfile=False):
        """ Computes the average order book profile, averaged over trading time, from the csv sourcefile. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean computation
        ----------
        args:
            num_levels_calc:    number of levels which should be considered for the output
            write_output:         if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (mean_bid, mean_ask)  in format of numpy arrays
        """

        print("Starting computation of average order book profile in file %s." % self.lobfilename)

        num_levels_calc = self.get_num_levels(num_levels_calc_str)

        tempval1 = 0.0
        tempval2 = 0.0
        comp = np.zeros(num_levels_calc * 2)  # compensator for lost low-order bits
        mean = np.zeros(num_levels_calc * 2)  # running mean

        with open(self.lobfilename + ".csv", newline='') as csvfile:
            lobdata = csv.reader(csvfile, delimiter=',')
            num_lines = sum(1 for row in lobdata)
            print("Loaded successfully. Number of lines: " + str(num_lines))
            csvfile.seek(0)  # reset iterator to beginning of the file
            print("Start calculation.")
            for row in lobdata:  # data are read as list of strings
                currorders = np.fromiter(row[1:(4 * num_levels_calc + 1):2],
                                         np.float)  # parse to integer
                for ctr, currorder in enumerate(currorders):
                    # print(lobstate)
                    tempval1 = currorder / num_lines - comp[ctr]
                    tempval2 = mean[ctr] + tempval1
                    comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                    mean[ctr] = tempval2

            print("Calculation finished.")

            # Add data to self.data
            self.add_data("--".join(("ttime-" + AV_ORDERBOOK_FILE_ID, "bid")), mean[1::2])
            self.add_data("--".join(("ttime-" + AV_ORDERBOOK_FILE_ID, "ask")), mean[0::2])

            if not write_outputfile:
                # LOBster format: bid data at odd * 2, LOBster format: ask data at even * 2
                return mean[1::2], mean[0::2]

            self._write_output_to_csv(num_levels_calc, mean[1::2], mean[0::2])
            return mean[1::2], mean[0::2]

    def variance_profile_tt(self, num_levels_calc_str="", write_outputfile=False):
        """ Computes the average order book profile, averaged over trading time, from the csv sourcefile. To avoid numerical errors by summing up large numbers, the Kahan Summation algorithm is used for mean computation
        ----------
        args:
            num_levels_calc:    number of levels which should be considered for the output
            write_output:       if True, then the average order book profile is stored as a csv file
        ----------
        output:
            (std_dev_bid, std_dev_ask)  in format of numpy arrays
        """

        print("Starting computation of average order book profile in file %s." % self.lobfilename)

        num_levels_calc = self.get_num_levels(num_levels_calc_str)
        # this is the mean of ask, mean of bid. Find mid price of these
        mean = self.average_profile_tt(num_levels_calc_str, write_outputfile)
        mid_mean = (mean[0] + mean[1]) / 2

        tempval1 = 0.0
        tempval2 = 0.0
        comp = np.zeros(num_levels_calc * 2)  # compensator for lost low-order bits
        variance = np.zeros(num_levels_calc * 2)  # running mean

        with open(self.lobfilename + ".csv", newline='') as csvfile:
            lobdata = csv.reader(csvfile, delimiter=',')
            num_lines = sum(1 for row in lobdata)
            print("Loaded successfully. Number of lines: " + str(num_lines))
            csvfile.seek(0)  # reset iterator to beginning of the file
            print("Start calculation.")
            for row in lobdata:  # data are read as list of strings
                currorders = np.fromiter(row[1:(4 * num_levels_calc + 1):2], np.float)  # parse to integer
                for ctr, currorder in enumerate(currorders):
                    # print(lobstate)
                    # use meanmidprice[ctr] here instead of mean[ctr]
                    tempval1 = (currorder / num_lines - mid_mean[np.int(ctr / 2)]) ** 2 - comp[ctr]
                    tempval2 = variance[ctr] + tempval1
                    comp[ctr] = (tempval2 - variance[ctr]) - tempval1
                    variance[ctr] = tempval2

            print("Calculation finished.")

            # Add data to self.data
            self.add_data("--".join(("ttime-" + AV_ORDERBOOK_FILE_ID, "bid")), mean[1::2])
            self.add_data("--".join(("ttime-" + AV_ORDERBOOK_FILE_ID, "ask")), mean[0::2])

            if not write_outputfile:
                # LOBster format: bid data at odd * 2, LOBster format: ask data at even * 2
                return variance[1::2], variance[0::2]

            self._write_output_to_csv(num_levels_calc, variance[1::2], variance[0::2])
            return variance[1::2], variance[0::2]

    def standardize_price_levels(self, num_levels_calc=""):
        """ Convert limit order book price levels from absolute prices to percentage of at-the-touch bid/ask price.
                ----------
                args:
                    num_levels_calc:    number of levels which should be considered for the output
                    write_output:       if True, then the average order book profile is stored as a csv file
                ----------
                output:
                    pandas
        """

        def _calc_price(price: float):
            if np.isnan(price) or np.abs(price) > 99999:
                return np.nan
            return price

        def _calc_volume(volume: float):
            if np.isnan(volume) or np.abs(volume) > 99999999:
                return np.nan
            return volume

        with open(self.lobfilename + ".csv", newline='') as orderbookfile:
            lobdata = csv.reader(orderbookfile, delimiter=',')
            rowLOB = next(lobdata)
            new_prices = np.array(len(rowLOB))
            for rowLOB in lobdata:
                currprofile = np.fromiter(rowLOB, np.float)  # parse to float, extract bucket volumes only at t(0)
                updated_prices = np.fromiter(currprofile[0:(4 * num_levels_calc) + 1:2], _calc_price)
                updated_sizes = np.fromiter(currprofile[1:(4 * num_levels_calc) + 1:2], _calc_volume)
                updated_values = np.empty((updated_prices.size + updated_sizes.size,), dtype=updated_prices.dtype)
                updated_values[0::2] = updated_prices
                updated_values[1::2] = updated_sizes
                new_prices.append(updated_values)
        return new_prices

    def _write_output_to_csv(self, num_levels_calc, **kwargs):
        print("Write output file.")
        outfilename = self.create_filestr("-".join(("ttime", AV_ORDERBOOK_FILE_ID)), str(num_levels_calc))
        outfilename = ".".join((outfilename, 'csv'))
        with open(outfilename, 'w') as outfile:
            wr = csv.writer(outfile)
            for arg in kwargs:
                wr.writerow(arg)
        print("Average order book saved as %s." % outfilename)

    def get_num_levels(self, num_levels_calc_str):
        num_levels_calc = self.num_levels
        if not (num_levels_calc_str == ""):
            num_levels_calc = int(num_levels_calc_str)
        if self.num_levels < num_levels_calc:
            raise DataRequestError(
                "Number of levels in data ({0}) is smaller than number of levels requested for calculation ({1}).".format(
                    self.num_levels, num_levels_calc))
        return num_levels_calc

    def average_profile(
            self,
            num_levels_calc_str="",
            write_outputfile=False
    ):
        """ Returns the average oder book profile from the csv sourcefile, averaged in real time. To avoid numerical
        errors by summing up large numbers, the Kahan Summation algorithm is used for mean computation """

        if num_levels_calc_str == "":
            num_levels_calc = self.num_levels_calc
        else:
            num_levels_calc = int(num_levels_calc_str)

        if int(self.num_levels) < num_levels_calc:
            raise DataRequestError(
                "Number of levels in data ({0}) is smaller than number of levels requested for calculation ({1}).".format(
                    self.num_level, num_levels_calc))

        time_start = float(self.time_start_calc / 1000.)
        time_end = float(self.time_end_calc / 1000.)
        mean = np.zeros(num_levels_calc * 2)  # running mean
        tempval1 = 0.0
        tempval2 = 0.0
        linectr = 0
        comp = np.zeros(num_levels_calc * 2)  # compensator for lost low-order bits
        flag = 0

        with open(".".join((self.lobfilename, 'csv')), newline='') as orderbookfile, open(
                ".".join((self.msgfilename, 'csv')), newline='') as messagefile:
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')

            rowMES = next(messagedata)  # data are read as list of strings
            rowLOB = next(lobdata)
            nexttime = float(rowMES[0])  # t(0)
            if time_end < nexttime:
                # In this case there are no entries in the file for the selected time interval. Array of 0s is returned
                warnings.warn(
                    "The first entry in the data files is after the end of the selected time period. Arrays of 0s "
                    "will be returned as mean.")
                return mean[1::2], mean[0::2]
            currprofile = np.fromiter(rowLOB[1:(4 * num_levels_calc + 1):2],
                                      np.float)  # parse to integer, extract bucket volumes only at t(0)
            if time_start <= nexttime:
                flag = 1

            for rowLOB, rowMES in zip(lobdata,
                                      messagedata):  # data are read as list of string, iterator now starts at second entry (since first has been exhausted above)
                currtime = nexttime  # (t(i))
                nexttime = float(rowMES[0])  # (t(i+1))
                if flag == 0:
                    if time_start <= nexttime:
                        # Start calculation
                        flag = 1
                        currtime = time_start

                        for ctr, currbucket in enumerate(currprofile):
                            tempval1 = (nexttime - currtime) / float(time_end - time_start) * currbucket - comp[ctr]
                            tempval2 = mean[ctr] + tempval1
                            comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                            mean[ctr] = tempval2
                else:
                    if time_end < nexttime:
                        # Finish calculation
                        nexttime = time_end

                    for ctr, currbucket in enumerate(currprofile):
                        # print(currprofile)
                        tempval1 = (nexttime - currtime) / float(time_end - time_start) * currbucket - comp[ctr]
                        tempval2 = mean[ctr] + tempval1
                        comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                        mean[ctr] = tempval2

                    if time_end == nexttime:
                        # Finish calculation
                        break

                ## Update order book to time t(i+1)
                currprofile = np.fromiter(rowLOB[1:(4 * num_levels_calc + 1):2],
                                          np.float)  # parse to integer, extract bucket volumes only
            else:  # executed only when not quitted by break, i.e. time_end >= time at end of file in this case we
                # extrapolate
                warnings.warn(
                    "Extrapolated order book data since time_end exceed time at end of the file by %f seconds." % (
                            time_end - nexttime))
                currtime = nexttime
                nexttime = time_end
                for ctr, currbucket in enumerate(currprofile):
                    # print(lobstate)
                    tempval1 = (nexttime - currtime) / (time_end - time_start) * currbucket - comp[ctr]
                    tempval2 = mean[ctr] + tempval1
                    comp[ctr] = (tempval2 - mean[ctr]) - tempval1
                    mean[ctr] = tempval2

        print("Calculation finished.")

        # Add data to self.data
        self.add_data("--".join((AV_ORDERBOOK_FILE_ID, "bid")), mean[1::2])
        self.add_data("--".join((AV_ORDERBOOK_FILE_ID, "ask")), mean[0::2])

        if not write_outputfile:
            return mean[1::2], mean[0::2]  # LOBster format: bid data at odd * 2,  LOBster format: ask data at even * 2

        print("Write output file.")
        outfilename = self.create_filestr(AV_ORDERBOOK_FILE_ID, str(num_levels_calc))
        outfilename = ".".join((outfilename, 'csv'))
        with open(outfilename, 'w') as outfile:
            wr = csv.writer(outfile)
            wr.writerow(mean[1::2])  # LOBster format: bid data at odd * 2
            wr.writerow(mean[0::2])  # LOBster format: ask data at even * 2

        print("Average order book saved as %s." % outfilename)
        return mean[1::2], mean[0::2]

    def _load_ordervolume(
            self,
            num_observations,
            num_levels_calc,
            profile2vol_fct=np.sum
    ):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean.

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array.
        '''

        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0  # counter for the outer of the
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep=True)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only
            currprofile = np.fromiter(rowLOB[1:(4 * num_levels_calc + 1):2], np.float)
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of volume in the file. Keep processes constant
                    if (ctr_obs > 0):
                        volume_bid[ctr_obs] = volume_bid[ctr_obs - 1]
                        volume_ask[ctr_obs] = volume_ask[ctr_obs - 1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.",
                                      RuntimeWarning)
                        volume_bid[ctr_obs] = 0.
                        volume_ask[ctr_obs] = 0.
                    continue

                while (time_stamp >= time_file):
                    # extract order volume from profile
                    volume_bid[ctr_obs] = profile2vol_fct(currprofile[1::2])
                    volume_ask[ctr_obs] = profile2vol_fct(currprofile[0::2])

                    # read next line
                    try:
                        rowMES = next(messagedata)  # data are read as list of strings
                        rowLOB = next(lobdata)
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update currprofile and time_file
                    currprofile = np.fromiter(rowLOB[1:(4 * num_levels_calc + 1):2],
                                              np.float)  # parse to integer, extract bucket volumes only
                    time_file = float(rowMES[0])

        if (file_ended_line < num_observations):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i" % (
                    num_observations - file_ended_line), RuntimeWarning)

        return dt, time_stamps, volume_bid, volume_ask

    def _load_ordervolume_levelx(
            self,
            num_observations,
            level
    ):
        ''' Extracts the volume of orders in the first num_level buckets at a uniform time grid of num_observations observations from the interval [time_start_calc, time_end_calc]. The volume process is extrapolated constantly on the last level in the file, for the case that time_end_calc is larger than the last time stamp in the file. profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean.

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array.
        '''

        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0  # counter for the outer of the
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep=True)
        volume_bid = np.zeros(num_observations)
        volume_ask = np.zeros(num_observations)

        # Ask level x is at position (x-1)*4 + 1, bid level x is at position (x-1)*4 + 3
        x_bid = (int(level) - 1) * 4 + 3
        x_ask = (int(level) - 1) * 4 + 1

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only

            # currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)
            currbid = float(rowLOB[x_bid])
            currask = float(rowLOB[x_ask])
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of volume in the file. Keep processes constant
                    if (ctr_obs > 0):
                        volume_bid[ctr_obs] = volume_bid[ctr_obs - 1]
                        volume_ask[ctr_obs] = volume_ask[ctr_obs - 1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.",
                                      RuntimeWarning)
                        volume_bid[ctr_obs] = 0.
                        volume_ask[ctr_obs] = 0.
                    continue

                while (time_stamp >= time_file):
                    # extract order volume from profile
                    volume_bid[ctr_obs] = currbid
                    volume_ask[ctr_obs] = currask

                    # read next line
                    try:
                        rowMES = next(messagedata)  # data are read as list of strings
                        rowLOB = next(lobdata)
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update currprofile and time_file
                    # currprofile = np.fromiter(rowLOB[1:(4*num_levels_calc + 1):2], np.float)    # parse to integer, extract bucket volumes only
                    currbid = float(rowLOB[x_bid])
                    currask = float(rowLOB[x_ask])
                    time_file = float(rowMES[0])

        if (file_ended_line < num_observations):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i" % (
                    num_observations - file_ended_line), RuntimeWarning)

        return dt, time_stamps, volume_bid, volume_ask

    def _load_ordervolume_full(
            self,
            num_levels_calc,
            profile2vol_fct=np.sum,
            ret_np=True
    ):
        ''' Extracts the volume of orders in the first num_level buckets from the interval [time_start_calc, time_end_calc].  profile2vol_fct allows to specify how the volume should be summarized from the profile. Typical choices are np.sum or np.mean. If ret_np==False then the output format are lists, else numpy arrays

        Note: Due to possibly large amount of data we iterate through the file instead of reading the whole file into an array.
        '''
        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.

        time_stamps = []
        volume_bid = []
        volume_ask = []
        index_start = -1
        index_end = -1

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings

            for ctrRow, (rowLOB, rowMES) in enumerate(zip(lobdata, messagedata)):
                time_now = float(rowMES[0])
                if (index_start == -1) and (time_now >= time_start_calc):
                    index_start = ctrRow
                if (index_end == -1) and (time_now > time_end_calc):
                    index_end = ctrRow
                    break

                time_stamps.append(time_now)
                currprofile = np.fromiter(rowLOB[1:(4 * num_levels_calc + 1):2],
                                          np.float)  # parse to integer, extract bucket volumes only
                volume_bid.append(profile2vol_fct(currprofile[1::2]))
                volume_ask.append(profile2vol_fct(currprofile[0::2]))

        if index_end == -1:
            # file end reached
            index_end = len(time_stamps)

        if ret_np:
            return np.array(time_stamps[index_start:index_end]), np.array(volume_bid[index_start:index_end]), np.array(
                volume_ask[index_start:index_end])
        return time_stamps[index_start:index_end], volume_bid[index_start:index_end], volume_ask[index_start:index_end]

    def _load_prices(
            self,
            num_observations
    ):
        ''' private method to implement how the price data are loaded from the files '''
        time_start_calc = float(self.time_start_calc) / 1000.
        time_end_calc = float(self.time_end_calc) / 1000.
        file_ended_line = int(num_observations)
        ctr_time = 0
        ctr_line = 0
        ctr_obs = 0  # counter for the outer of the
        time_stamps, dt = np.linspace(time_start_calc, time_end_calc, num_observations, retstep=True)
        prices_bid = np.empty(num_observations)
        prices_ask = np.empty(num_observations)

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            time_file = float(rowMES[0])

            for ctr_obs, time_stamp in enumerate(time_stamps):
                if (time_stamp < time_file):
                    # no update of prices in the file. Keep processes constant
                    if (ctr_obs > 0):
                        prices_bid[ctr_obs] = prices_bid[ctr_obs - 1]
                        prices_ask[ctr_obs] = prices_ask[ctr_obs - 1]
                    else:
                        # so far no data available, raise warning and set processes to 0.
                        warnings.warn("Data do not contain beginning of the monitoring period. Values set to 0.",
                                      RuntimeWarning)
                        prices_bid[ctr_obs] = 0.
                        prices_ask[ctr_obs] = 0.
                    continue

                while (time_stamp >= time_file):
                    # LOBster stores best ask and bid price in resp. 1st and 3rd column, price in unit USD*10000
                    prices_bid[ctr_obs] = float(rowLOB[2]) / float(10000)
                    prices_ask[ctr_obs] = float(rowLOB[0]) / float(10000)

                    # read next line
                    try:
                        rowMES = next(messagedata)  # data are read as list of strings
                        rowLOB = next(lobdata)
                    except StopIteration:
                        if (file_ended_line == num_observations):
                            file_ended_line = ctr_obs
                        break
                    # update time_file
                    time_file = float(rowMES[0])

        if (file_ended_line < num_observations - 1):
            warnings.warn("End of file reached. Number of values constantly extrapolated: %i" % (
                    num_observations - file_ended_line), RuntimeWarning)
            while ctr_obs < (num_observations - 1):
                prices_bid[ctr_obs + 1] = prices_bid[ctr_obs]
                prices_ask[ctr_obs + 1] = prices_ask[ctr_obs]

        return dt, time_stamps, prices_bid, prices_ask

    def _load_profile_snapshot_lobster(
            self,
            time_stamp,
            num_levels_calc=None
    ):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        '''
        # convert time from msec to sec
        time_stamp = float(time_stamp) / 1000.

        if num_levels_calc is None:
            num_levels_calc = self.num_levels_calc

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)
            # parse to float, extract bucket volumes only
            time_file = float(rowMES[0])
            if time_file > time_stamp:
                raise LookupError(
                    "Time data in the file start at {} which is after time_stamps: {}".format(time_file, time_stamp))
            if time_file == time_stamp:
                # file format is [ask level, ask volume, bid level, bid volume, ask level, ....]
                # conversion of price levels to USD
                bid_prices = np.fromiter(rowLOB[2:(4 * num_levels_calc):4], np.float) / float(10000)
                bid_volume = np.fromiter(rowLOB[3:(4 * num_levels_calc):4], np.float)
                # conversion of price levels to USD
                ask_prices = np.fromiter(rowLOB[0:(4 * num_levels_calc):4], np.float) / float(10000)
                ask_volume = np.fromiter(rowLOB[1:(4 * num_levels_calc):4], np.float)

            for rowMES in messagedata:
                time_file = float(rowMES[0])
                if time_file > time_stamp:
                    # file format is [ask level, ask volume, bid level, bid volume, ask level, ....]
                    # conversion of price levels to USD
                    bid_prices = np.fromiter(rowLOB[2:(4 * num_levels_calc):4], np.float) / float(10000)
                    bid_volume = np.fromiter(rowLOB[3:(4 * num_levels_calc):4], np.float)
                    # conversion of price levels to USD
                    ask_prices = np.fromiter(rowLOB[0:(4 * num_levels_calc):4], np.float) / float(10000)
                    ask_volume = np.fromiter(rowLOB[1:(4 * num_levels_calc):4], np.float)
                    break

                rowLOB = next(lobdata)
            else:
                # time in file did not exceed time stamp to the end. Return last entries of the file
                bid_prices = np.fromiter(rowLOB[2:(4 * num_levels_calc):4], np.float) / float(10000)
                bid_volume = np.fromiter(rowLOB[3:(4 * num_levels_calc):4], np.float)
                # conversion of price levels to USD
                ask_prices = np.fromiter(rowLOB[0:(4 * num_levels_calc):4], np.float) / float(10000)
                ask_volume = np.fromiter(rowLOB[1:(4 * num_levels_calc):4], np.float)
            return bid_prices, bid_volume, ask_prices, ask_volume

    def load_profile_snapshot(
            self,
            time_stamp,
            num_levels_calc=None
    ):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        '''
        return self._load_profile_snapshot_lobster(time_stamp, num_levels_calc)

    def get_start_and_end_bell_tickers(self, include_quote_starts: bool, num_levels_calc=None):
        ''' Returns a two numpy arrays with snapshots of the bid- and ask-side of the order book at a given time stamp
                Output:
                bid_prices, bid_volume, ask_prices, ask_volume
        '''
        if num_levels_calc is None:
            num_levels_calc = self.num_levels_calc

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = csv.reader(orderbookfile, delimiter=',')
            messagedata = csv.reader(messagefile, delimiter=',')
            # get first row
            # data are read as list of strings
            rowMES = next(messagedata)
            rowLOB = next(lobdata)

            time_finish = float(rowMES[-2])

            for rowMES in messagedata:
                if int(rowMES[1]) == 7 and int(rowMES[5]) == -1:
                    if int(rowMES[4]) == 0 and not include_quote_starts:
                        continue
                    rowMES_next = next(rowMES)
                    if float(rowMES_next[0]) > time_finish:
                        raise DataRequestError("Found an invalid timestamp")
                    bid_prices, bid_volume, ask_prices, ask_volume = self.load_profile_snapshot(rowMES_next[0])
                    print(bid_prices, bid_volume, ask_prices, ask_volume)

    def log_prices_over_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        """Returns log returns for time period involved
        Output:
        standard deviation (volatility) of log-returns"""
        bid_prices, bid_volumes, ask_prices, ask_volumes = self.select_orders_within_time(
            time_offset_ms, time_period_ms, num_levels_calc=num_levels_calc)
        mid_prices = np.fromiter((_calc_mid_price(bid[0], ask[0]) for bid, ask in zip(bid_prices, ask_prices)),
                                 np.float)
        mid_prices = np.ma.masked_equal(mid_prices, 0)
        mid_prices = mid_prices.compressed()[~np.isnan(mid_prices.compressed())]
        sod_price = list(mid_prices)[0]
        return np.fromiter((np.log(x / sod_price) for x in mid_prices), np.float)

    def volume_at_touch_over_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        times, bid_prices, bid_volumes, ask_prices, ask_volumes = self.select_orders_within_time(
            time_offset_ms, time_period_ms, num_levels_calc=num_levels_calc)

        return times, bid_volumes, ask_volumes

    def cumulative_notional_over_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        """Returns cumulative notionals for time period involved
                Output:
                two pandas dataframes - one with mid prices, one with bid-ask spreads"""

        times, bid_prices, bid_volumes, ask_prices, ask_volumes = self.select_orders_within_time(
            time_offset_ms, time_period_ms, num_levels_calc=num_levels_calc)

        dataset_period = pd.DataFrame(data=np.array([times, bid_prices, bid_volumes, ask_prices, ask_volumes]),
                                      columns=['Time', 'Bid Prices', 'Bid Volumes', 'Ask Prices', 'Ask Volumes',
                                               'Buy Intensity', 'Sell Intensity'])

        mid_prices = np.fromiter((_calc_mid_price(bid[0], ask[0]) for bid, ask in zip(bid_prices, ask_prices)),
                                 np.float)
        mid_prices = np.ma.masked_equal(mid_prices, 0)
        mid_prices = mid_prices.compressed()[~np.isnan(mid_prices.compressed())]

        bid_notional = [np.fromiter((price[0] * qty[0] for price, qty in zip(bid_prices, bid_volumes)), np.float)]
        ask_notional = [np.fromiter((price[0] * qty[0] for price, qty in zip(ask_prices, ask_volumes)), np.float)]

        for i in range(1, num_levels_calc):
            ask_notional.append(np.fromiter((price[i] * qty[i] for price, qty in zip(ask_prices, ask_volumes)),
                                            np.float) + ask_notional[-1])
            bid_notional.append(np.fromiter((price[i] * qty[i] for price, qty in zip(bid_prices, bid_volumes)),
                                            np.float) + bid_notional[-1])

        bid_ntnl_data = pd.DataFrame(data=np.array(bid_notional), columns=["Cumulative Bid Notional Level " + str(i) for i in num_levels_calc])
        ask_ntnl_data = pd.DataFrame(data=np.array(ask_notional), columns=["Cumulative Ask Notional Level " + str(i) for i in num_levels_calc])
        bid_data_joined = dataset_period.join(bid_ntnl_data)
        return bid_data_joined.join(ask_ntnl_data)

    def batch_spreads_over_time(self, time_offset_ms: float, time_period_ms: float, interval_ms: float,
                                num_levels_calc=None):
        return batch_data(self.spreads_over_time(time_offset_ms, time_period_ms, num_levels_calc),
                          lambda x: x, interval_ms, time_offset_ms, time_period_ms)

    def spreads_over_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        """Returns log returns for time period involved
                Output:
                two pandas dataframes - one with mid prices, one with bid-ask spreads"""

        times, bid_prices, bid_volumes, ask_prices, ask_volumes = self.select_orders_within_time(
            time_offset_ms, time_period_ms, num_levels_calc=num_levels_calc)

        dataset_period = pd.DataFrame(data=np.array([times, bid_prices, bid_volumes, ask_prices, ask_volumes]),
                                      columns=['Time', 'Bid Prices', 'Bid Volumes', 'Ask Prices', 'Ask Volumes',
                                               'Buy Intensity', 'Sell Intensity'])

        mid_prices = np.fromiter((_calc_mid_price(bid[0], ask[0]) for bid, ask in zip(bid_prices, ask_prices)),
                                 np.float)
        mid_prices = np.ma.masked_equal(mid_prices, 0)
        mid_prices = mid_prices.compressed()[~np.isnan(mid_prices.compressed())]

        spreads = np.fromiter((_calc_spread(bid[0], ask[0]) for bid, ask in zip(bid_prices, ask_prices)),
                              np.float)
        spreads = np.ma.masked_equal(spreads, 0)
        spreads = spreads.compressed()[~np.isnan(spreads.compressed())]

        dataset_period['Mid Prices'] = mid_prices
        dataset_period['Spread'] = spreads

        return dataset_period

    def batch_log_returns_in_time(self, time_offset_ms: float, time_period_ms: float, interval_ms: float,
                                  num_levels_calc=None):
        return batch_data(self.log_returns_over_time(time_offset_ms, time_period_ms, num_levels_calc),
                          _calc_returns, interval_ms, time_offset_ms, time_period_ms)

    def log_returns_over_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        """Returns log returns for time period involved
        Output:
        standard deviation (volatility) of log-returns"""

        dataset_period = self.mid_prices_over_time(num_levels_calc, time_offset_ms, time_period_ms)
        mid_prices = dataset_period['Mid Prices']
        shifted_prices = [mid_prices[0]] + list(mid_prices)[1:]
        dataset_period['Shifted Prices'] = shifted_prices
        dataset_period['Log Returns'] = [np.log(a, b) for a, b in zip(mid_prices, shifted_prices)]

        return dataset_period

    def mid_prices_over_time(self, num_levels_calc: float, time_offset_ms: float, time_period_ms: float):
        times, bid_prices, bid_volumes, ask_prices, ask_volumes = self.select_orders_within_time(
            time_offset_ms, time_period_ms, num_levels_calc=num_levels_calc)
        dataset_period = pd.DataFrame(data=np.array([times, bid_prices, bid_volumes, ask_prices, ask_volumes]),
                                      columns=['Time', 'Bid Prices', 'Bid Volumes', 'Ask Prices', 'Ask Volumes',
                                               'Buy Intensity', 'Sell Intensity'])
        mid_prices = np.fromiter((_calc_mid_price(bid[0], ask[0]) for bid, ask in zip(bid_prices, ask_prices)),
                                 np.float)
        mid_prices = np.ma.masked_equal(mid_prices, 0)
        mid_prices = mid_prices.compressed()[~np.isnan(mid_prices.compressed())]
        dataset_period['Mid Prices'] = mid_prices
        return dataset_period

    def select_orders_within_time(self, time_offset_ms: float, time_period_ms: float, num_levels_calc=None):
        """ Returns a four numpy arrays with message and orderbook data of the bid- and ask-side of the order book
        at a given time stamp
        Output:
        bid_prices, bid_volume, ask_prices, ask_volume
        """
        if num_levels_calc is None:
            num_levels_calc = self.num_levels_calc

        with open((self.lobfilename + '.csv')) as orderbookfile, open(self.msgfilename + '.csv') as messagefile:
            # Read data from csv file
            lobdata = pd.read_csv(orderbookfile, sep=',')
            messagedata = pd.read_csv(messagefile, sep=',',
                                      names=['Time', 'Event_Type', 'Order_Id', 'Size', 'Price', 'Direction', 'NaN'])

            start_time = lobdata['Time'].iloc[0]

            # Direction is marked as sell (-1), since a buy market order liquidates a sell LO
            buy_mo = messagedata.loc[(messagedata['Direction'] == -1)
                                     & ((messagedata['Event_Type'] == 4) or (messagedata['Event_Type'] == 5))]
            buy_intensity = [index / (time - start_time) for index, time in enumerate(buy_mo['Time'])]
            # Direction is marked as buy (+1), since a sell market order liquidates a buy LO
            sell_mo = messagedata.loc[(messagedata['Direction'] == 1)
                                      & ((messagedata['Event_Type'] == 4) or (messagedata['Event_Type'] == 5))]
            sell_intensity = [index / (time - start_time) for index, time in enumerate(sell_mo['Time'])]

            time_file = float(messagedata[messagedata.columns[0]][0])
            time_start = time_file + time_offset_ms
            time_stop = time_start + time_period_ms
            lobdata = pd.DataFrame(messagedata[messagedata.columns[0]]).join(lobdata)
            cols = ['Time']
            for i in range(self.num_levels * 4):
                bidask = 'Bid' if np.floor(i / 2) % 2 else 'Ask'
                size = 'Size' if i % 2 else ''
                cols.append('_'.join((bidask, str(np.floor(i / 4) + 1), size)))
            lobdata.columns = cols
            messagedata = messagedata.loc[(messagedata['Time'] <= time_stop) & (messagedata['Time'] >= time_start)]
            lobdata = lobdata.loc[(lobdata['Time'] <= time_stop) & (lobdata['Time'] >= time_start)]

            # conversion of price levels to USD
            bid_prices = np.array(lobdata[lobdata.columns[3]].apply(lambda x: x / float(10000)))
            bid_volumes = np.array(lobdata[lobdata.columns[4]])
            # conversion of price levels to USD
            ask_prices = np.array(lobdata[lobdata.columns[1]].apply(lambda x: x / float(10000)))
            ask_volumes = np.array(lobdata[lobdata.columns[2]])
            times = np.array(messagedata[messagedata.columns[0]])
            return times, bid_prices, bid_volumes, ask_prices, ask_volumes, buy_intensity, sell_intensity

    def calc_normalization_vals(self):
        prev_day_reader = LOBSTERReader(self.ticker_str, _prev_workday(self.date_str, HOLUSD()).__str__(),
                                        str(self.time_start), self.time_end_str, self.num_levels_str)
        mean_bid, mean_ask = prev_day_reader.average_profile_tt()
        mean = np.mean((mean_bid[0], mean_ask[0]))
        std_dev = 1

        return mean, std_dev
# END LOBSTERReader
