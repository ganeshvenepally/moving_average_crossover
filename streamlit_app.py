023-06-30 02:58:34.162 Uncaught app exception

Traceback (most recent call last):

  File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 552, in _run_script

    exec(code, module.__dict__)

  File "/app/moving_average_crossover/streamlit_app.py", line 75, in <module>

    portfolio, stock_df, signals = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)

  File "/app/moving_average_crossover/streamlit_app.py", line 11, in MovingAverageCrossStrategy

    stock_df = stock.history(start=start_date, end=end_date)['Close']

  File "/home/appuser/venv/lib/python3.9/site-packages/yfinance/utils.py", line 105, in wrapper

    result = func(*args, **kwargs)

  File "/home/appuser/venv/lib/python3.9/site-packages/yfinance/base.py", line 153, in history

    tz = self._get_ticker_tz(proxy, timeout)

  File "/home/appuser/venv/lib/python3.9/site-packages/yfinance/base.py", line 1422, in _get_ticker_tz

    cache.store(self.ticker, tz)

  File "/home/appuser/venv/lib/python3.9/site-packages/yfinance/utils.py", line 984, in store

    raise Exception("Tkr {} tz already in cache".format(tkr))

Exception: Tkr AAPL tz already in cache

2023-06-30 02:58:34.447 Uncaught app exception

Traceback (most recent call last):

  File "/home/appuser/venv/lib/python3.9/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 552, in _run_script

    exec(code, module.__dict__)

  File "/app/moving_average_crossover/streamlit_app.py", line 75, in <module>

    portfolio, stock_df, signals = MovingAverageCrossStrategy(stock_symbol, start_date, end_date, short_window, long_window, moving_avg, display_table, initial_cash)

  File "/app/moving_average_crossover/streamlit_app.py", line 55, in MovingAverageCrossStrategy

    portfolio["Shares"] = [portfolio.loc[row, "Signal"] == "Buy" and 1 or portfolio.loc[row, "Signal"] == "Sell" and 0 or portfolio["Shares"][row - 1] for row in np.arange(len(portfolio))]

  File "/app/moving_average_crossover/streamlit_app.py", line 55, in <listcomp>

    portfolio["Shares"] = [portfolio.loc[row, "Signal"] == "Buy" and 1 or portfolio.loc[row, "Signal"] == "Sell" and 0 or portfolio["Shares"][row - 1] for row in np.arange(len(portfolio))]

  File "/home/appuser/venv/lib/python3.9/site-packages/pandas/core/indexing.py", line 1096, in __getitem__

    return self.obj._get_value(*key, takeable=self._takeable)

  File "/home/appuser/venv/lib/python3.9/site-packages/pandas/core/frame.py", line 3877, in _get_value

    row = self.index.get_loc(index)

  File "/home/appuser/venv/lib/python3.9/site-packages/pandas/core/indexes/datetimes.py", line 581, in get_loc

    raise KeyError(key)

KeyError: 0
