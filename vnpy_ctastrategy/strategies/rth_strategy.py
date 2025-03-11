from copy import copy
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import talib
from rqdatac.utils import convert_bar_to_multi_df
from scipy.stats import linregress

from vnpy.trader.constant import CandleType, BarDowTheory, Direction, Interval, Offset, TrendType
from vnpy.trader.database import DB_TZ
from vnpy.trader.utility import ceil_to, floor_to, round_to
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


class RTHStrategy(CtaTemplate):
    """"""

    author = "用Python的交易员"

    bar_window     = 5

    # parameters
    vqi_period     = 5
    vqi_smoothing  = 2
    vqi_filter     = 2
    vqi_ma_method  = 3 # 3 = MODE_LWMA
    currency_point = 1 # 1 = 1点
    pricetick      = 5.0
    stop_loss      = 20.0

    trailing_start = 100.0
    trailing_point = 20.0


    # variables
    vqi = 0.0
    prev_vqi = 0.0
    trend = TrendType.SIDE.value
    close_price = 0.0
    sma5_close = 0.0
    enable_open = False # 是否允许开仓, 只有在新的bar才允许开仓
    INTER_TIME = timedelta(seconds=5)  # 交易时间间隔 5秒
    close_time: datetime = None  # 平仓时间
    close_tm = None # 平仓时间


    parameters = ["vqi_period", "vqi_smoothing", "stop_loss"]
    variables = ["vqi", "prev_vqi", "trend", "close_price", "sma5_close", "close_tm"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, self.bar_window, self.on_xmin_bar)
        self.am = ArrayManager()
        self.prev_bar = None
        self.cur_bar = None
        self.interval = Interval.MINUTE5
        self.process_time = datetime.now(DB_TZ)
        self.vqi_data: Dict[int, float] = {}

        # self.trailing_stop_long  = TrailingStopLong(trailing_point=self.trailing_point)
        # self.trailing_stop_short = TrailingStopShort(trailing_point=self.trailing_point)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10, interval=self.interval, callback=self.on_bar_process)

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
        bg = self.bg
        bg.update_tick(tick)
        bar: BarData = None
        if self.bar_window > 0:
            # Update 1 minute bar into generator
            bg.update_bar(bg.bar)
            bar = copy(bg.window_bar)
        else:
            bar = copy(bg.bar)
        bar.datetime = bar.datetime.replace(second=0, microsecond=0)
        new_minute = False
        if (
            (self.cur_bar.datetime.minute != bar.datetime.minute)
            or (self.cur_bar.datetime.hour != bar.datetime.hour)
        ):
            new_minute = True
        self.on_bar_process(bar, new_minute, update_gb=False)


    def on_bar_process(self, bar: BarData, new_minute: bool = True, update_gb: bool = True):
        """
        Callback of new bar data update.
        """
        # Update 1 minute bar into x minute window(load bar from history)
        if update_gb and self.bar_window > 0:
            self.bg.update_bar(bar)

        if new_minute:
            self.prev_vqi      = self.vqi
            self.prev_bar      = self.cur_bar
            self.close_time   = bar.datetime + timedelta(minutes=self.bar_window) - self.INTER_TIME
            self.close_tm     = self.close_time.strftime("%H:%M:%S")
            self.enable_open = True
            bar_time = bar.datetime.strftime("%H:%M:%S")
            # 日本股市开市时间 8:45-9:15 12:30-12:45 15:35-17:15停止交易
            # 美国经济指标发布时间 21:30-21:45 22:30-22:45 停止交易
            # 美国股市开市时间 23:30-23:45 停止交易
            if ("09:00:00" <= bar_time < "09:15:00") or (
                    "12:30:00" <= bar_time < "12:45:00") or (
                    "15:35:00" <= bar_time < "17:15:00") or (
                    "21:30:00" <= bar_time < "21:45:00") or (
                    "22:30:00" <= bar_time < "22:45:00") or (
                    "23:30:00" <= bar_time < "23:45:00"):
                self.enable_open = False
            self.write_log(f"new_minute: {bar.datetime}")
        self.cur_bar = bar

        am = self.am
        am.update_bar(bar, new_minute)
        if not am.inited:
            return

        # 取得最后一个bar的vqi值
        pre_vqi = 0.0
        self.vqi = self.caculate_vqi(self.vqi_smoothing)

        # trend
        trend = TrendType.SIDE
        # 上一个bar的上影线超过35，判定为下降趋势
        if self.vqi > 0:
            trend = TrendType.UP
        elif self.prev_vqi < 0 and self.is_hammer(self.prev_bar):
            trend = TrendType.UP
        elif self.vqi < 0:
            trend = TrendType.DOWN
        self.trend = trend.value
        # close price
        close_price = round_to(bar.close_price, self.pricetick)
        self.close_price = close_price
        # sma5 close
        sma5 = talib.SMA(am.close, timeperiod=3)
        self.sma5_close = round_to(sma5[-1], self.pricetick)
        if trend == TrendType.UP:
            self.sma5_close = min(close_price, round_to(sma5[-1], self.pricetick))
        elif trend == TrendType.DOWN:
            self.sma5_close = max(close_price, round_to(sma5[-1], self.pricetick))


        if datetime.now(DB_TZ) - self.process_time <= timedelta(milliseconds=200):
            self.put_event()
            return
        self.process_time = datetime.now(DB_TZ)


        if self.trading:
            # 强制止损线
            if self.pos < 0 and close_price >= self.short_price + self.stop_loss:
                canceled = self.cancel_no_target_orders(close_price, Direction.LONG, Offset.CLOSE)
                if canceled:
                    exist = self.exist_target_orders(close_price, Direction.LONG, Offset.CLOSE)
                    if not exist:
                        self.write_log(f"stop_loss cover(buy): {close_price}")
                        self.cover(close_price, 1)
                        self.enable_open = False

            if self.pos > 0 and close_price <= self.long_price - self.stop_loss:
                canceled = self.cancel_no_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
                if canceled:
                    exist = self.exist_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
                    if not exist:
                        self.write_log(f"stop_loss sell: {close_price}")
                        self.sell(close_price, 1)
                        self.enable_open = False

            # 到达每个bar的结束时间，用close_price先平仓
            if self.process_time >= self.close_time or new_minute:
                if self.pos > 0:
                    canceled = self.cancel_no_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
                    if canceled:
                        exist = self.exist_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
                        if not exist:
                            self.write_log(f"time_close sell: {close_price}")
                            self.sell(close_price, 1)
                            self.enable_open = False

                elif self.pos < 0:
                    canceled = self.cancel_no_target_orders(close_price, Direction.LONG, Offset.CLOSE)
                    if canceled:
                        exist = self.exist_target_orders(close_price, Direction.LONG, Offset.CLOSE)
                        if not exist:
                            self.write_log(f"time_close cover(buy): {close_price}")
                            self.cover(close_price, 1)
                            self.enable_open = False

            # # 移动止盈线
            # if self.pos < 0:
            #     if close_price <= self.short_price - self.trailing_start:
            #         if not self.trailing_stop_short.started:
            #             self.write_log(f"trailing_stop_short start: {close_price}")
            #             self.trailing_stop_short.enter_trade(close_price)
            #     if not self.trailing_stop_short.update_price(close_price):
            #         canceled = self.cancel_no_target_orders(close_price, Direction.LONG, Offset.CLOSE)
            #         if canceled:
            #             exist = self.exist_target_orders(close_price, Direction.LONG, Offset.CLOSE)
            #             if not exist:
            #                 self.write_log(f"trailing_stop cover(buy): {close_price}")
            #                 self.cover(close_price, 1)
            #
            #
            # if self.pos > 0:
            #     if close_price >= self.long_price + self.trailing_start:
            #         if not self.trailing_stop_long.started:
            #             self.write_log(f"trailing_stop_long start: {close_price}")
            #             self.trailing_stop_long.enter_trade(close_price)  # トレーリングストップ開始
            #     if not self.trailing_stop_long.update_price(close_price):  # ポジション終了
            #         canceled = self.cancel_no_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
            #         if canceled:
            #             exist = self.exist_target_orders(close_price, Direction.SHORT, Offset.CLOSE)
            #             if not exist:
            #                 self.write_log(f"trailing_stop sell: {close_price}")
            #                 self.sell(close_price, 1)


            # 等待信号有趋势时候再开仓
            if self.pos == 0 and self.enable_open:
                price_match = False
                # buy
                if trend == TrendType.UP:
                    # 信号转变为买，买开仓
                    # 取消不是目标价位的活跃订单
                    canceled = self.cancel_no_target_orders(self.sma5_close, Direction.LONG, Offset.OPEN)
                    if canceled:
                        exist = self.exist_target_orders(self.sma5_close, Direction.LONG, Offset.OPEN)
                        if not exist:
                            self.write_log(f"buy: {self.sma5_close}")
                            self.buy(self.sma5_close, 1)
                            self.long_price = self.sma5_close
                            price_match = True
                            # self.trailing_stop_long.started = False
                            # self.trailing_stop_short.started = False

                # sell
                elif trend == TrendType.DOWN:
                    # 信号转变为卖，卖开仓
                    # 取消不是目标价位的活跃订单
                    canceled = self.cancel_no_target_orders(self.sma5_close, Direction.SHORT, Offset.OPEN)
                    if canceled:
                        exist = self.exist_target_orders(self.sma5_close, Direction.SHORT, Offset.OPEN)
                        if not exist:
                            self.write_log(f"short: {self.sma5_close}")
                            self.short(self.sma5_close, 1)
                            self.short_price = self.sma5_close
                            price_match = True
                            # self.trailing_stop_long.started = False
                            # self.trailing_stop_short.started = False

                # no signal
                else:
                    self.cancel_all()

                if not price_match:
                    if close_price > self.prev_bar.high_price:
                        canceled = self.cancel_no_target_orders(close_price, Direction.LONG, Offset.OPEN)
                        if canceled:
                            exist = self.exist_target_orders(close_price, Direction.LONG, Offset.OPEN)
                            if not exist:
                                self.write_log(f"breakout buy: {close_price}")
                                self.buy(close_price, 1)
                                self.long_price = close_price

                    elif close_price < self.prev_bar.low_price:
                        canceled = self.cancel_no_target_orders(close_price, Direction.SHORT, Offset.OPEN)
                        if canceled:
                            exist = self.exist_target_orders(close_price, Direction.SHORT, Offset.OPEN)
                            if not exist:
                                self.write_log(f"breakout short: {close_price}")
                                self.short(close_price, 1)
                                self.short_price = close_price

        # 更新UI界面
        self.put_event()


    def cancel_no_target_orders(self, price: float, direction: Direction, offset: Offset) -> bool:
        """"""
        no_target_orders = self.get_target_active_orders(price, direction, offset, target=False)
        if len(no_target_orders) != 0:
            self.cancel_orders(no_target_orders)
            # 还有活跃订单，取消还未成功
            return False
        else:
            # 没有活跃订单，取消已经成功
            return True

    def exist_target_orders(self, price: float, direction: Direction, offset: Offset) -> bool:
        """"""
        target_orders = self.get_target_active_orders(price, direction, offset, target=True)
        if len(target_orders) == 0:
            # 没有目标价位的活跃订单
            return False
        else:
            # 存在目标价位的活跃订单
            return True

    # get active target order ids
    def get_target_active_orders(self, price: float, direction: Direction, offset: Offset, target: bool=True) -> list:
        vt_orderids: set = self.cta_engine.strategy_orderid_map.get(self.strategy_name, set())
        if len(vt_orderids) == 0:
            return []

        vt_ids = []
        for vt_orderid in copy(vt_orderids):
            order: Optional[OrderData] = self.cta_engine.main_engine.get_order(vt_orderid)
            if not order:
                continue
            if target:
                if order.price == price and order.direction == direction and order.offset == offset:
                    vt_ids.append(vt_orderid)
            else:
                if order.price != price or order.direction != direction or order.offset != offset:
                    vt_ids.append(vt_orderid)
        return vt_ids

    def on_bar(self, bar: BarData, new_minute: bool = True):
        """
        Callback of new bar data.
        """

    def on_xmin_bar(self, bar: BarData):
        """"""
        # chart: ChartWidget = self.charts[bar.vt_symbol]
        # chart.update_bar(bar)


    def caculate_vqi(self, smoothing: float) -> float:
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
        c2 = maC_array[-1-smoothing]
        max_p = max(h - l, max(h - c2, c2 - l))
        if (max_p != 0 and (h - l) != 0):
            VQ = abs(((c - c2) / max_p + (c - o) / (h - l)) * 0.5) * ((c - c2 + (c - o)) * 0.5)
            # vqi = pre_vqi if abs(VQ) < self.vqi_filter * self.currency_point else VQ
            vqi = VQ
            return vqi
        else:
            return 0.0

    def is_hammer(self, bar: BarData) -> bool:
        """
        判断是否锤子线
        """
        # full_len = bar.high_price - bar.low_price
        # if full_len == 0:
        #     return False

        # 计算实体部分、高低影线
        body = abs(bar.close_price - bar.open_price)
        lower_shadow = min(bar.close_price, bar.open_price) - bar.low_price
        upper_shadow = bar.high_price - max(bar.close_price, bar.open_price)

        # shadow_high = min(bar.close_price, bar.open_price)
        # shadow_low  = bar.low_price
        # # 计算前一根 K 线的实体范围
        # prev_body_high = max(prev_bar.close_price, prev_bar.open_price)
        # prev_body_low = min(prev_bar.close_price, prev_bar.open_price)

        # 判断 Hammer 条件：
        hammer_condition = (
                (5.0 <= body <= 15.0) and # 实体长度在 5-15 点之间
                (lower_shadow >= 3 * body) and  # 下影线至少是实体的 3 倍
                (upper_shadow < 0.3 * lower_shadow)  # 上影线很短
        )

        # # 判断下影线与前一根 K 线实体部分的重叠是否较少（< 30%）
        # overlap = max(0, min(shadow_high, prev_body_high) - max(shadow_low, prev_body_low))
        # prev_body_range = prev_body_high - prev_body_low if prev_body_high != prev_body_low else 1  # 避免除零
        # low_overlap_condition = (overlap / prev_body_range) < 0.3  # 重叠部分小于前一根 K 线实体 30%

        return hammer_condition

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


class TrailingStopLong:
    def __init__(self, trailing_point: float):
        """
        :param trailing_point: トレーリングストップのpoint（例: 10.0）
        """
        self.trailing_point = trailing_point
        self.entry_price = None
        self.stop_price = None
        self.highest_price = None
        self.started = False

    def enter_trade(self, price: float):
        """エントリー時に価格を設定"""
        self.entry_price = price
        self.highest_price = price
        self.stop_price = price - self.trailing_point
        self.started = True
        print(f"Entered long trade at {price}, initial stop set to {self.stop_price:.2f}")

    def update_price(self, current_price: float):
        """価格更新に基づいてトレーリングストップを調整"""
        if not self.started:
            return True # トレーリングストップ未開始

        if self.entry_price is None:
            raise ValueError("Trade not entered yet!")

        if current_price > self.highest_price:
            self.highest_price = current_price
            self.stop_price = current_price - self.trailing_point
            print(f"New high: {current_price}, stop moved to {self.stop_price:.2f}")

        if current_price <= self.stop_price:
            print(f"Stop long loss hit at {self.stop_price:.2f}. Exiting trade.")
            return False  # ポジション終了
        return True  # 継続


class TrailingStopShort:
    def __init__(self, trailing_point: float):
        """
        :param trailing_point: トレーリングストップのpoint（例: 10.0）
        """
        self.trailing_point = trailing_point
        self.entry_price = None
        self.stop_price = None
        self.lowest_price = None
        self.started = False

    def enter_trade(self, price: float):
        """エントリー時に価格を設定"""
        self.entry_price = price
        self.lowest_price = price
        self.stop_price = price + self.trailing_point
        self.started = True
        print(f"Entered short trade at {price}, initial stop set to {self.stop_price:.2f}")

    def update_price(self, current_price: float):
        """価格更新に基づいてトレーリングストップを調整"""
        if not self.started:
            return True # トレーリングストップ未開始

        if self.entry_price is None:
            raise ValueError("Trade not entered yet!")

        if current_price < self.lowest_price:
            self.lowest_price = current_price
            self.stop_price = current_price + self.trailing_point
            print(f"New low: {current_price}, stop moved to {self.stop_price:.2f}")

        if current_price >= self.stop_price:
            print(f"Stop short loss hit at {self.stop_price:.2f}. Exiting trade.")
            return False  # ポジション終了
        return True  # 継続
