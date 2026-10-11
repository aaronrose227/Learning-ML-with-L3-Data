"""Turn a Nasdaq TotalView-ITCH 5.0 file into LOBSTER-style tables for a few tickers.

    python scripts/extract_itch.py data/12302019.NASDAQ_ITCH50.gz --tickers AAPL MSFT

For each ticker this writes two Parquet files whose rows line up one-to-one:

<TICKER>_<DATE>_messages.parquet -- one row per order book event (the L3 feed)
    time      nanoseconds since midnight, exchange time (US Eastern)
    event     add | cancel | delete | execute | hidden_execute | cross | halt | resume
    order_id  exchange order reference, 0 where the event has no visible order
    side      +1 resting buy order, -1 resting sell order, 0 not reported
    price     price level of the order affected, in units of $0.0001
    size      shares added, removed or traded by this event
    itch      letter of the original ITCH message

<TICKER>_<DATE>_book.parquet -- the visible book straight after that row's event
    ask_price_1, ask_size_1, bid_price_1, bid_size_1, ask_price_2, ...
    prices in units of $0.0001 (NaN where a level is empty), sizes in shares

How the ITCH messages map onto `event`:
    A, F  add             a new visible limit order
    X     cancel          part of an order is cancelled; size is the shares removed
    D     delete          the whole order is removed; size is what was left of it
    E, C  execute         a visible order trades. side is the resting order's side,
                          so side=+1 means the trade was seller-initiated. price is
                          the order's limit price, which for C can differ from the
                          price it traded at (mostly in the auctions).
    U     delete + add    a replace, written as two rows with the same time
    P     hidden_execute  a trade against a non-displayed order; the book does not
                          change and the feed does not report the side
    Q     cross           an opening or closing auction print; price is the auction
                          price and size the total shares
    H     halt / resume   written only when trading in the stock stops or restarts
"""

import argparse
import gzip
import re
import struct
import time
from array import array
from bisect import bisect_left, insort
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = ["add", "cancel", "delete", "execute", "hidden_execute", "cross", "halt", "resume"]
ADD, CANCEL, DELETE, EXECUTE, HIDDEN, CROSS, HALT, RESUME = range(len(EVENTS))

# Every message starts with: type (1 byte), stock locate (2), tracking number (2),
# timestamp (6). These decode what follows that 11-byte header.
HEADER = 11
ADD_BODY = struct.Struct(">QcI8sI")  # order ref, side, shares, ticker, price
REDUCE_BODY = struct.Struct(">QI")  # order ref, shares
DELETE_BODY = struct.Struct(">Q")  # order ref
REPLACE_BODY = struct.Struct(">QQII")  # old ref, new ref, shares, price
CROSS_BODY = struct.Struct(">Q8sI")  # shares, ticker, price

CHUNK = 1 << 24


class Book:
    """Live orders and depth for one ticker, plus the rows recorded so far."""

    def __init__(self, levels):
        self.levels = levels
        self.orders = {}  # order ref -> [side, price, shares left]
        self.depth = {1: {}, -1: {}}  # side -> {price: total visible shares}
        self.prices = {1: [], -1: []}  # side -> sorted prices that have depth
        # Best `levels` on each side as [price, size, price, size, ...], 0 = empty.
        self.top = {1: [0] * (2 * levels), -1: [0] * (2 * levels)}
        self.halted = False
        self.unknown = 0  # messages that referenced an order we never saw

        self.time = array("q")
        self.event = array("b")
        self.order_id = array("q")
        self.side = array("b")
        self.price = array("q")
        self.size = array("q")
        self.itch = array("B")
        self.bid = array("i")
        self.ask = array("i")

    def record(self, ts, event, order_id, side, price, size, itch):
        self.time.append(ts)
        self.event.append(event)
        self.order_id.append(order_id)
        self.side.append(side)
        self.price.append(price)
        self.size.append(size)
        self.itch.append(itch)
        self.bid.extend(self.top[1])
        self.ask.extend(self.top[-1])

    def _refresh(self, side):
        depth = self.depth[side]
        prices = self.prices[side]
        best = prices[-self.levels :][::-1] if side == 1 else prices[: self.levels]
        top = []
        for price in best:
            top.append(price)
            top.append(depth[price])
        top.extend([0] * (2 * self.levels - len(top)))
        self.top[side] = top

    def add(self, ts, order_id, side, shares, price, itch):
        self.orders[order_id] = [side, price, shares]
        depth = self.depth[side]
        if price in depth:
            depth[price] += shares
        else:
            depth[price] = shares
            insort(self.prices[side], price)
        self._refresh(side)
        self.record(ts, ADD, order_id, side, price, shares, itch)

    def remove(self, ts, event, order_id, shares, itch):
        """Take `shares` off a resting order (all of it if None). Returns its side."""
        order = self.orders.get(order_id)
        if order is None:
            self.unknown += 1
            return None
        side, price, left = order
        if shares is None or shares >= left:
            shares = left
            del self.orders[order_id]
        else:
            order[2] = left - shares

        depth = self.depth[side]
        remaining = depth[price] - shares
        if remaining:
            depth[price] = remaining
        else:
            del depth[price]
            prices = self.prices[side]
            del prices[bisect_left(prices, price)]
        self._refresh(side)
        self.record(ts, event, order_id, side, price, shares, itch)
        return side

    def frames(self):
        letters = np.frombuffer(self.itch, dtype=np.uint8).view("S1").astype(str)
        messages = pd.DataFrame(
            {
                "time": np.frombuffer(self.time, dtype=np.int64),
                "event": pd.Categorical.from_codes(
                    np.frombuffer(self.event, dtype=np.int8), EVENTS
                ),
                "order_id": np.frombuffer(self.order_id, dtype=np.int64),
                "side": np.frombuffer(self.side, dtype=np.int8),
                "price": np.frombuffer(self.price, dtype=np.int64),
                "size": np.frombuffer(self.size, dtype=np.int64),
                "itch": pd.Categorical(letters),
            }
        )

        width = 2 * self.levels
        sides = {
            "ask": np.frombuffer(self.ask, dtype=np.intc).reshape(-1, width),
            "bid": np.frombuffer(self.bid, dtype=np.intc).reshape(-1, width),
        }
        columns = {}
        for level in range(self.levels):
            for name, top in sides.items():
                price = top[:, 2 * level].astype(np.float64)
                price[price == 0] = np.nan
                columns[f"{name}_price_{level + 1}"] = price
                columns[f"{name}_size_{level + 1}"] = top[:, 2 * level + 1]
        return messages, pd.DataFrame(columns)


def extract(path, tickers, levels, limit=None):
    """Replay the file and return ({ticker: Book}, session events, messages read)."""
    wanted = {ticker.encode().ljust(8): ticker for ticker in tickers}
    books = {}  # stock locate -> Book
    names = {}  # stock locate -> ticker
    session = []  # (time, system event code)
    count = 0
    started = time.time()

    with gzip.open(path, "rb") as f:
        buf = b""
        pos = 0
        while limit is None or count < limit:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            buf = buf[pos:] + chunk
            pos = 0
            end = len(buf)
            before = count // 50_000_000

            while end - pos >= 2:
                start = pos + 2
                stop = start + ((buf[pos] << 8) | buf[pos + 1])
                if stop > end:
                    break
                pos = stop
                count += 1

                kind = buf[start]
                locate = (buf[start + 1] << 8) | buf[start + 2]
                book = books.get(locate)
                if book is None:
                    if kind == 82:  # R: stock directory, gives each ticker's locate
                        ticker = wanted.get(buf[start + 11 : start + 19])
                        if ticker is not None:
                            books[locate] = Book(levels)
                            names[locate] = ticker
                    elif kind == 83:  # S: start/end of the trading session
                        ts = int.from_bytes(buf[start + 5 : start + 11], "big")
                        session.append((ts, chr(buf[start + 11])))
                    continue

                ts = int.from_bytes(buf[start + 5 : start + 11], "big")
                body = start + HEADER
                if kind == 65 or kind == 70:  # A, F
                    ref, side, shares, _, price = ADD_BODY.unpack_from(buf, body)
                    book.add(ts, ref, 1 if side == b"B" else -1, shares, price, kind)
                elif kind == 68:  # D
                    (ref,) = DELETE_BODY.unpack_from(buf, body)
                    book.remove(ts, DELETE, ref, None, kind)
                elif kind == 85:  # U
                    old, new, shares, price = REPLACE_BODY.unpack_from(buf, body)
                    side = book.remove(ts, DELETE, old, None, kind)
                    if side is not None:
                        book.add(ts, new, side, shares, price, kind)
                elif kind == 69 or kind == 67:  # E, C
                    ref, shares = REDUCE_BODY.unpack_from(buf, body)
                    book.remove(ts, EXECUTE, ref, shares, kind)
                elif kind == 88:  # X
                    ref, shares = REDUCE_BODY.unpack_from(buf, body)
                    book.remove(ts, CANCEL, ref, shares, kind)
                elif kind == 80:  # P
                    _, _, shares, _, price = ADD_BODY.unpack_from(buf, body)
                    book.record(ts, HIDDEN, 0, 0, price, shares, kind)
                elif kind == 81:  # Q
                    shares, _, price = CROSS_BODY.unpack_from(buf, body)
                    if shares:
                        book.record(ts, CROSS, 0, 0, price, shares, kind)
                elif kind == 72:  # H: trading state, 'T' means trading normally
                    halted = buf[start + 19] != 84
                    if halted != book.halted:
                        book.halted = halted
                        book.record(ts, HALT if halted else RESUME, 0, 0, 0, 0, kind)

            if count // 50_000_000 != before:
                print(f"  {count / 1e6:,.0f}M messages read, {time.time() - started:,.0f}s")

    return {names[locate]: book for locate, book in books.items()}, session, count


def clock(ns):
    seconds = ns // 1_000_000_000
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("path", type=Path, help="ITCH 5.0 file, gzipped")
    parser.add_argument(
        "--tickers", nargs="+", default=["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
    )
    parser.add_argument("--levels", type=int, default=10, help="book levels to keep")
    parser.add_argument("--out", type=Path, default=None, help="default: next to the input")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default: from the file name")
    parser.add_argument("--limit", type=int, default=None, help="stop after about this many messages")
    args = parser.parse_args()

    date = args.date
    if date is None:
        match = re.match(r"(\d{2})(\d{2})(\d{4})", args.path.name)
        if match is None:
            parser.error("cannot read the date from the file name, pass --date")
        date = f"{match[3]}-{match[1]}-{match[2]}"
    out = args.out or args.path.parent
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.path} for {', '.join(args.tickers)}")
    books, session, count = extract(args.path, args.tickers, args.levels, args.limit)
    print(f"Read {count:,} messages")
    print("Session events: " + ", ".join(f"{code} {clock(ts)}" for ts, code in session))

    for ticker in args.tickers:
        book = books.get(ticker)
        if book is None:
            print(f"{ticker}: not found in the file")
            continue
        messages, depth = book.frames()
        messages.to_parquet(out / f"{ticker}_{date}_messages.parquet", compression="zstd", index=False)
        depth.to_parquet(out / f"{ticker}_{date}_book.parquet", compression="zstd", index=False)
        print(f"{ticker}: {len(messages):,} rows, {book.unknown:,} references to unknown orders")


if __name__ == "__main__":
    main()
