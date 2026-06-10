#!/usr/bin/env node
'use strict';

const path = require('node:path');
require('dotenv').config({ path: path.resolve(__dirname, '..', '.env') });

const C = require('./config');
const {
  TradingView,
  authOptions,
  requireAuth,
  parseDate,
  unix,
  fetchHistory,
  fetchIndicator,
  writeCsv,
  writeJson,
  mergeCloseSeries,
} = require('./lib');

function parseArgs(argv) {
  const result = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      result._.push(token);
      continue;
    }
    const [rawKey, inline] = token.slice(2).split('=', 2);
    if (inline !== undefined) {
      result[rawKey] = inline;
    } else if (argv[i + 1] && !argv[i + 1].startsWith('--')) {
      result[rawKey] = argv[i + 1];
      i += 1;
    } else {
      result[rawKey] = true;
    }
  }
  return result;
}

function symbolId(value) {
  return C.MARKETS[value] || value;
}

function selectedMarkets(value) {
  if (!value) return C.MARKETS;
  const selected = {};
  value.split(',').forEach((item) => {
    const key = item.trim();
    selected[key] = symbolId(key);
  });
  return selected;
}

function json(value) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function log(message, quiet = false) {
  if (!quiet) process.stderr.write(`${message}\n`);
}

async function history(args) {
  const symbol = symbolId(args.symbol || args._[1] || 'BTC');
  const alias = args.alias || args.symbol || args._[1] || 'BTC';
  const result = await fetchHistory({
    symbol,
    timeframe: args.timeframe || C.DEFAULT_TIMEFRAME,
    start: args.start || C.DEFAULT_START,
    end: args.end,
    range: args.range ? Number(args.range) : undefined,
    timeoutMs: args.timeout ? Number(args.timeout) * 1000 : undefined,
  });
  const output = args.output || path.join(C.RAW_DIR, `${alias}.csv`);
  writeCsv(output, result.periods, ['date', 'time', 'open', 'high', 'low', 'close', 'volume']);
  log(`Saved ${result.periods.length} bars for ${symbol} -> ${output}`, args.quiet);
  return result;
}

async function historyAll(args) {
  const start = args.start || C.DEFAULT_START;
  const end = args.end || new Date().toISOString().slice(0, 10);
  const timeframe = args.timeframe || C.DEFAULT_TIMEFRAME;
  const markets = selectedMarkets(args.symbols);
  const series = {};
  const errors = {};

  for (const [alias, symbol] of Object.entries(markets)) {
    log(`Fetching ${alias} (${symbol})...`, args.quiet);
    try {
      const result = await fetchHistory({
        symbol,
        timeframe,
        start,
        end,
        range: args.range ? Number(args.range) : undefined,
        timeoutMs: args.timeout ? Number(args.timeout) * 1000 : undefined,
      });
      series[alias] = result;
      writeCsv(
        path.join(C.RAW_DIR, `${alias}.csv`),
        result.periods,
        ['date', 'time', 'open', 'high', 'low', 'close', 'volume'],
      );
      log(`  ${result.periods.length} bars`, args.quiet);
    } catch (error) {
      errors[alias] = error.message;
      log(`  skipped: ${error.message}`, args.quiet);
    }
  }

  if (!Object.keys(series).length) throw new Error('TradingView returned no market data');

  const rows = mergeCloseSeries(series);
  const aliases = Object.keys(series);
  const output = args.output || path.join(C.RAW_DIR, 'prices.csv');
  writeCsv(output, rows, ['date', ...aliases]);
  writeJson(path.join(C.RAW_DIR, 'metadata.json'), {
    fetchedAt: new Date().toISOString(),
    start,
    end,
    timeframe,
    output,
    markets,
    observations: Object.fromEntries(
      Object.entries(series).map(([alias, result]) => [alias, result.periods.length]),
    ),
    errors,
  });
  log(`Saved ${rows.length} dates x ${aliases.length} markets -> ${output}`, args.quiet);
  return { rows, series, errors, output };
}

async function search(args) {
  if (args.indicator) {
    const results = await TradingView.searchIndicator(args.indicator);
    json(results.slice(0, Number(args.limit || 20)).map((item) => ({
      id: item.id,
      version: item.version,
      name: item.name,
      type: item.type,
      access: item.access,
      author: item.author,
    })));
    return;
  }
  const query = args.market || args._[1] || 'BTCUSDT';
  const results = await TradingView.searchMarketV3(query, args.type || '');
  json(results.slice(0, Number(args.limit || 20)).map((item) => ({
    id: item.id,
    exchange: item.exchange,
    symbol: item.symbol,
    description: item.description,
    type: item.type,
  })));
}

async function technicalAnalysis(args) {
  const symbol = symbolId(args.symbol || args._[1] || 'BTC');
  json({ symbol, advice: await TradingView.getTA(symbol) });
}

async function indicator(args) {
  const indicatorId = args.id || args._[1];
  if (!indicatorId) throw new Error('Specify --id STD;RSI or another Pine indicator ID');
  const options = args.options ? JSON.parse(args.options) : {};
  const result = await fetchIndicator({
    symbol: symbolId(args.symbol || 'BTC'),
    indicatorId,
    timeframe: args.timeframe || C.DEFAULT_TIMEFRAME,
    range: Number(args.range || 300),
    options,
    builtIn: Boolean(args.builtin || indicatorId.includes('@tv-')),
  });
  const output = args.output || path.join(C.RAW_DIR, `indicator-${indicatorId.replaceAll(';', '-')}.json`);
  writeJson(output, result);
  log(`Saved indicator data -> ${output}`, args.quiet);
  if (!args.quiet) json(result.indicator);
}

async function privateIndicators() {
  requireAuth('Private indicators');
  const results = await TradingView.getPrivateIndicators(
    process.env.TV_SESSION,
    process.env.TV_SIGNATURE,
  );
  json(results.map(({ id, version, name, type, access }) => ({
    id, version, name, type, access,
  })));
}

async function drawings(args) {
  const layout = args.layout || args._[1];
  if (!layout) throw new Error('Specify --layout <TradingView layout ID>');
  let credentials = {};
  if (process.env.TV_SESSION) {
    const user = await TradingView.getUser(process.env.TV_SESSION, process.env.TV_SIGNATURE || '');
    credentials = {
      id: user.id,
      session: process.env.TV_SESSION,
      signature: process.env.TV_SIGNATURE || '',
    };
  }
  json(await TradingView.getDrawings(layout, args.symbol || '', credentials, args.chart || '_shared'));
}

function openStreams(symbols, onTick, durationSeconds = 0) {
  return new Promise((resolve, reject) => {
    const client = new TradingView.Client(authOptions());
    const charts = [];
    let timer;
    let closing = false;

    async function close() {
      if (closing) return;
      closing = true;
      clearTimeout(timer);
      charts.forEach((chart) => chart.delete());
      await client.end();
      resolve();
    }

    client.onError((...parts) => {
      if (closing) return;
      reject(new Error(parts.map(String).join(' ')));
    });

    for (const [alias, symbol] of Object.entries(symbols)) {
      const chart = new client.Session.Chart();
      charts.push(chart);
      let lastKey = '';
      chart.onError((...parts) => log(`${alias}: ${parts.map(String).join(' ')}`));
      chart.onUpdate(async () => {
        const period = chart.periods[0];
        if (!period) return;
        const key = `${period.time}:${period.close}`;
        if (key === lastKey) return;
        lastKey = key;
        await onTick({
          alias,
          symbol,
          receivedAt: new Date().toISOString(),
          ...period,
          high: period.max,
          low: period.min,
        }, close);
      });
      chart.setMarket(symbol, { timeframe: '1', range: 1 });
    }

    if (durationSeconds > 0) timer = setTimeout(close, durationSeconds * 1000);
    process.once('SIGINT', close);
    process.once('SIGTERM', close);
  });
}

async function live(args) {
  const markets = selectedMarkets(args.symbols || args.symbol || 'BTC');
  await openStreams(markets, async (tick) => {
    process.stdout.write(`${JSON.stringify(tick)}\n`);
  }, Number(args.duration || 0));
}

async function spread(args) {
  const markets = selectedMarkets(args.symbols || 'BINANCE:BTCUSDT,COINBASE:BTCUSD');
  const prices = {};
  await openStreams(markets, async (tick) => {
    prices[tick.alias] = tick.close;
    if (Object.keys(prices).length !== Object.keys(markets).length) return;
    const values = Object.entries(prices);
    const low = values.reduce((a, b) => (a[1] < b[1] ? a : b));
    const high = values.reduce((a, b) => (a[1] > b[1] ? a : b));
    process.stdout.write(`${JSON.stringify({
      receivedAt: tick.receivedAt,
      prices,
      low: { market: low[0], price: low[1] },
      high: { market: high[0], price: high[1] },
      spread: high[1] - low[1],
      spreadPct: ((high[1] / low[1]) - 1) * 100,
    })}\n`);
  }, Number(args.duration || 0));
}

async function sendAlert(message) {
  const jobs = [];
  if (process.env.TELEGRAM_BOT_TOKEN && process.env.TELEGRAM_CHAT_ID) {
    jobs.push(fetch(`https://api.telegram.org/bot${process.env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ chat_id: process.env.TELEGRAM_CHAT_ID, text: message }),
    }));
  }
  if (process.env.DISCORD_WEBHOOK_URL) {
    jobs.push(fetch(process.env.DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ content: message }),
    }));
  }
  await Promise.all(jobs);
}

async function alert(args) {
  const symbol = args.symbol || 'BTC';
  const above = args.above === undefined ? null : Number(args.above);
  const below = args.below === undefined ? null : Number(args.below);
  if (above === null && below === null) throw new Error('Specify --above <price> or --below <price>');

  await openStreams(selectedMarkets(symbol), async (tick, close) => {
    const hit = (above !== null && tick.close >= above) || (below !== null && tick.close <= below);
    if (!hit) return;
    const message = `${tick.symbol}: ${tick.close} crossed ${above !== null ? `above ${above}` : `below ${below}`}`;
    log(message);
    await sendAlert(message);
    await close();
  }, Number(args.duration || 0));
}

async function fakeReplay(args) {
  const start = args.start || new Date(Date.now() - 7 * 86400000).toISOString();
  const count = Number(args.count || 20);
  const timeframe = args.timeframe || '60';
  const startDate = parseDate(start);
  const minuteSize = Number(timeframe) || 1440;
  const end = new Date(startDate.getTime() + count * minuteSize * 60000).toISOString();
  const result = await fetchHistory({
    symbol: symbolId(args.symbol || 'BTC'),
    timeframe,
    start,
    end,
    range: count + 10,
  });
  for (const period of result.periods.slice(0, count)) {
    process.stdout.write(`${JSON.stringify(period)}\n`);
    if (args.interval) await new Promise((resolve) => setTimeout(resolve, Number(args.interval)));
  }
}

async function replay(args) {
  if (!args.real) return fakeReplay(args);
  requireAuth('Real replay mode');

  const client = new TradingView.Client(authOptions());
  const chart = new client.Session.Chart();
  const count = Number(args.count || 20);
  let seen = 0;
  let started = false;

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Replay timed out')), Number(args.timeout || 60) * 1000);
    chart.onError((...parts) => reject(new Error(parts.map(String).join(' '))));
    chart.onReplayLoaded(async () => {
      if (started) return;
      started = true;
      await chart.replayStart(Number(args.interval || 250));
    });
    chart.onUpdate(async () => {
      if (!started || !chart.periods[0]) return;
      process.stdout.write(`${JSON.stringify(chart.periods[0])}\n`);
      seen += 1;
      if (seen >= count) {
        clearTimeout(timeout);
        await chart.replayStop();
        resolve();
      }
    });
    chart.setMarket(symbolId(args.symbol || 'BTC'), {
      timeframe: args.timeframe || '60',
      replay: unix(args.start || new Date(Date.now() - 7 * 86400000).toISOString()),
      range: 1,
    });
  });

  chart.delete();
  await client.end();
}

function help() {
  process.stdout.write(`TradingView collector

Commands:
  history       Export one OHLCV series to CSV
  history-all   Export the configured research universe and merged prices.csv
  live          Stream realtime quotes as JSONL
  search        Search markets or Pine indicators
  ta            Get TradingView technical-analysis recommendations
  indicator     Export Pine/built-in indicator values
  private       List indicators available to your account
  replay        Replay historical bars (--real uses TradingView replay)
  spread        Stream price spreads across markets
  alert         Send a Telegram/Discord threshold alert
  drawings      Export drawings from a TradingView layout

Examples:
  npm run tv -- history-all --start 2019-01-01
  npm run tv -- search --market BTCUSDT
  npm run tv -- search --indicator RSI
  npm run tv -- ta --symbol BTC
  npm run tv -- indicator --id "STD;RSI" --symbol BTC --range 500
  npm run tv -- indicator --id "Volume@tv-basicstudies-241" --builtin --symbol BTC
  npm run tv -- live --symbols BTC,ETH --duration 10
  npm run tv -- spread --symbols BINANCE:BTCUSDT,COINBASE:BTCUSD --duration 10
`);
}

async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const command = args._[0] || 'help';
  const commands = {
    help,
    history,
    'history-all': historyAll,
    search,
    ta: technicalAnalysis,
    indicator,
    private: privateIndicators,
    drawings,
    live,
    spread,
    alert,
    replay,
  };
  if (!commands[command]) throw new Error(`Unknown command: ${command}`);
  return commands[command](args);
}

if (require.main === module) {
  main().catch((error) => {
    let message = error.message;
    if (/maximum number of studies|not auth|subscription/i.test(message)) {
      message += ' (try setting TV_SESSION and TV_SIGNATURE in .env for your TradingView account)';
    }
    process.stderr.write(`TradingView error: ${message}\n`);
    process.exitCode = 1;
  });
}

module.exports = { parseArgs, symbolId, selectedMarkets, main };
