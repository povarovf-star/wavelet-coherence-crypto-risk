'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const { parseArgs, symbolId, selectedMarkets } = require('../cli');
const {
  approximateRange,
  normalizePeriods,
  mergeCloseSeries,
  toCsv,
} = require('../lib');

test('parses CLI flags and positional values', () => {
  assert.deepEqual(parseArgs(['history', '--symbol', 'BTC', '--quiet', '--range=10']), {
    _: ['history'],
    symbol: 'BTC',
    quiet: true,
    range: '10',
  });
});

test('resolves configured aliases and raw TradingView symbols', () => {
  assert.equal(symbolId('BTC'), 'BINANCE:BTCUSDT');
  assert.equal(symbolId('COINBASE:BTCUSD'), 'COINBASE:BTCUSD');
  assert.deepEqual(selectedMarkets('BTC,COINBASE:BTCUSD'), {
    BTC: 'BINANCE:BTCUSDT',
    'COINBASE:BTCUSD': 'COINBASE:BTCUSD',
  });
});

test('normalizes, filters, and sorts chart periods', () => {
  const periods = normalizePeriods([
    { time: 1704153600, open: 2, max: 4, min: 1, close: 3, volume: 20 },
    { time: 1704067200, open: 1, max: 3, min: 0, close: 2, volume: 10 },
  ], '2024-01-01', '2024-01-01');
  assert.equal(periods.length, 1);
  assert.deepEqual(periods[0], {
    time: 1704067200,
    date: '2024-01-01',
    open: 1,
    high: 3,
    low: 0,
    close: 2,
    volume: 10,
  });
});

test('merges close series on an outer date join', () => {
  const rows = mergeCloseSeries({
    BTC: { periods: [{ date: '2024-01-01', close: 10 }] },
    SP500: { periods: [{ date: '2024-01-02', close: 20 }] },
  });
  assert.deepEqual(rows, [
    { date: '2024-01-01', BTC: 10 },
    { date: '2024-01-02', SP500: 20 },
  ]);
});

test('writes valid simple CSV and estimates ranges', () => {
  assert.equal(toCsv([{ a: 'hello, world', b: 2 }], ['a', 'b']), 'a,b\n"hello, world",2\n');
  assert.equal(approximateRange('2024-01-01', '2024-01-10', 'D'), 29);
});
