'use strict';

const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');

const MARKETS = {
  BTC: 'BINANCE:BTCUSDT',
  ETH: 'BINANCE:ETHUSDT',
  SOL: 'BINANCE:SOLUSDT',
  BNB: 'BINANCE:BNBUSDT',
  XRP: 'BINANCE:XRPUSDT',
  USDC: 'BINANCE:USDCUSDT',
  DAI: 'KRAKEN:DAIUSDT',
  SP500: 'SP:SPX',
  NASDAQ: 'NASDAQ:IXIC',
  GOLD: 'AMEX:GLD',
  DXY: 'TVC:DXY',
};

module.exports = {
  ROOT,
  MARKETS,
  DEFAULT_START: '2019-01-01',
  DEFAULT_TIMEFRAME: 'D',
  RAW_DIR: path.join(ROOT, 'data', 'raw', 'tradingview'),
};
