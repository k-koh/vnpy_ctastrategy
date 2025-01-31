from typing import Dict

import numpy as np
import talib
from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)


class VolatilityQualityStrategy(CtaTemplate):
    """"""

    author = "用Python的交易员"

    # parameters
    vqi_period = 5
    vqi_smoothing = 2
    vqi_filter = 1
    vqi_ma_method = 3 # 3 = MODE_LWMA

    fast_window = 10
    slow_window = 20

    # variables
    vqi = 0.0

    fast_ma0 = 0.0
    fast_ma1 = 0.0

    slow_ma0 = 0.0
    slow_ma1 = 0.0

    parameters = ["vqi_period", "vqi_smoothing", "vqi_filter", "vqi_ma_method"]
    variables = ["vqi", "fast_ma0", "fast_ma1", "slow_ma0", "slow_ma1"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

        self.currency_point = 1
        self.vqi_ma_method  = 3 # 3 = MODE_LWMA
        self.vqi_period     = 5
        self.vqi_smoothing  = 2
        self.vqi_filter     = 1
        self.vqi_data: Dict[int, float] = {}
        self.vqi_start      = self.vqi_smoothing + self.vqi_period + 3


    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        ix = am.count - 1
        self.vqi = self.get_vqi_value(ix)


        # buy
        if self.vqi > 0.0:
            if self.pos == 0:
                self.buy(bar.close_price, 1)
            elif self.pos < 0:
                self.cover(bar.close_price, 1)
                self.buy(bar.close_price, 1)
        # sell
        else:
            if self.pos == 0:
                self.short(bar.close_price, 1)
            elif self.pos > 0:
                self.sell(bar.close_price, 1)
                self.short(bar.close_price, 1)

        self.put_event()

    def get_vqi_value(self, ix: int) -> float:
        """"""
        # Return default value when no enough MA and smoothing data
        if ix < self.vqi_start:
            return 0

        # When initialize, calculate all vqi value
        am = self.am
        if not self.vqi_data:
            base_ix = 0
            open_data  = am.open
            high_data  = am.high
            low_data   = am.low
            close_data = am.close
            # initialize VQ with 0
            self.vqi_data = {n: 0.0 for n in range(len(open_data))}
            self.caculate_vqi(open_data, high_data, low_data, close_data, base_ix)

        # Calculate new value
        # and Recalculate current bar
        max_key = max(self.vqi_data.keys())
        if ix not in self.vqi_data or ix == max_key:
            size = self.vqi_start + 1
            open_data = np.zeros(size)
            high_data = np.zeros(size)
            low_data = np.zeros(size)
            close_data = np.zeros(size)
            base_ix = ix - self.vqi_start
            open_data[0:size] = am.open[-size:]
            high_data[0:size] = am.high[-size:]
            low_data[0:size] = am.low[-size:]
            close_data[0:size] = am.close[-size:]
            self.caculate_vqi(open_data, high_data, low_data, close_data, base_ix)

        if ix in self.vqi_data:
            return self.vqi_data[ix]


    def caculate_vqi(self, open_data: np.ndarray, high_data: np.ndarray, low_data: np.ndarray, close_data: np.ndarray, base_ix: int = 0):
        """"""
        maO_array = talib.MA(open_data,  timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maH_array = talib.MA(high_data,  timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maL_array = talib.MA(low_data,   timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maC_array = talib.MA(close_data, timeperiod=self.vqi_period, matype=self.vqi_ma_method)

        count = len(maO_array)
        # loop index [start, total - 1]
        for i in range(self.vqi_start, count):
            # calculate VQI
            o = maO_array[i]
            h = maH_array[i]
            l = maL_array[i]
            c = maC_array[i]
            c2 = maC_array[i - self.vqi_smoothing]
            max_p = max(h - l, max(h - c2, c2 - l))
            if (max_p != 0 and (h - l) != 0):
                VQ = abs(((c-c2)/max_p+(c-o)/(h-l))*0.5)*((c-c2+(c-o))*0.5)
                # self.vqi_data[base_ix+i] = self.vqi_data[base_ix+i-1] if abs(VQ) < self.vqi_filter * self.currency_point else VQ
                self.vqi_data[base_ix+i] = VQ

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
