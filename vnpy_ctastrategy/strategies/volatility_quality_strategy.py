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
    vqi_period     = 5
    vqi_smoothing  = 2
    vqi_filter     = 1
    vqi_ma_method  = 3 # 3 = MODE_LWMA
    currency_point = 1 # 1 = 1点

    # variables
    vqi = 0.0

    parameters = ["vqi_period", "vqi_smoothing", "vqi_filter", "vqi_ma_method", "currency_point"]
    variables = ["vqi"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()
        self.pre_vqi = 0.0

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

        self.vqi = self.get_vqi_value()
        self.pre_vqi = self.vqi

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

    def get_vqi_value(self) -> float:
        """"""
        maO_array = talib.MA(self.am.open,  timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maH_array = talib.MA(self.am.high,  timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maL_array = talib.MA(self.am.low,   timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        maC_array = talib.MA(self.am.close, timeperiod=self.vqi_period, matype=self.vqi_ma_method)
        # calculate VQI
        o = maO_array[-1]
        h = maH_array[-1]
        l = maL_array[-1]
        c = maC_array[-1]
        c2 = maC_array[-1-self.vqi_smoothing]
        max_p = max(h - l, max(h - c2, c2 - l))
        if (max_p != 0 and (h - l) != 0):
            VQ = abs(((c - c2) / max_p + (c - o) / (h - l)) * 0.5) * ((c - c2 + (c - o)) * 0.5)
            vqi = self.pre_vqi if abs(VQ) < self.vqi_filter * self.currency_point else VQ
            return vqi
        else:
            return 0.0

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
