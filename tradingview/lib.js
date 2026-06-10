'use strict';

const fs = require('node:fs');
const path = require('node:path');
const TradingView = require('@mathieuc/tradingview');

function authOptions() {
  if (!process.env.TV_SESSION) return {};
  return {
    token: process.env.TV_SESSION,
    signature: process.env.TV_SIGNATURE || '',
  };
}

function requireAuth(feature) {
  if (!process.env.TV_SESSION || !process.env.TV_SIGNATURE) {
    throw new Error(
      `${feature} requires TV_SESSION and TV_SIGNATURE from your own TradingView account`,
    );
  }
}

function parseDate(value, fallback = null) {
  if (!value) return fallback;
  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) throw new Error(`Invalid date: ${value}`);
  return new Date(timestamp);
}

function unix(value) {
  return Math.floor(parseDate(value).getTime() / 1000);
}

function isoDate(timestamp) {
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
}

function approximateRange(start, end, timeframe) {
  const from = parseDate(start);
  const to = parseDate(end, new Date());
  const seconds = Math.max(1, (to - from) / 1000);
  const unitSeconds = {
    D: 86400,
    W: 604800,
    M: 2592000,
  };
  const perBar = unitSeconds[timeframe] || Number(timeframe) * 60 || 86400;
  return Math.ceil(seconds / perBar) + 20;
}

function normalizePeriods(periods, start, end) {
  const from = start ? unix(start) : -Infinity;
  const to = end ? unix(end) + 86399 : Infinity;
  return periods
    .filter((p) => p.time >= from && p.time <= to)
    .sort((a, b) => a.time - b.time)
    .map((p) => ({
      time: p.time,
      date: isoDate(p.time),
      open: p.open,
      high: p.max,
      low: p.min,
      close: p.close,
      volume: p.volume,
    }));
}

function waitForChart(chart, options = {}) {
  const {
    minimum = 1,
    timeoutMs = 30000,
    quietMs = 800,
  } = options;

  return new Promise((resolve, reject) => {
    let quietTimer;
    const timeout = setTimeout(() => {
      reject(new Error(`TradingView chart timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    function done() {
      clearTimeout(timeout);
      clearTimeout(quietTimer);
      resolve(chart.periods);
    }

    chart.onError((...parts) => {
      clearTimeout(timeout);
      clearTimeout(quietTimer);
      reject(new Error(parts.map(String).join(' ')));
    });

    chart.onUpdate(() => {
      if (chart.periods.length < minimum) return;
      clearTimeout(quietTimer);
      quietTimer = setTimeout(done, quietMs);
    });
  });
}

async function fetchHistory({
  symbol,
  timeframe = 'D',
  start,
  end,
  range,
  timeoutMs = 45000,
}) {
  const client = new TradingView.Client(authOptions());
  const chart = new client.Session.Chart();
  const wanted = range || approximateRange(start, end, timeframe);

  try {
    chart.setMarket(symbol, {
      timeframe,
      range: wanted,
      to: end ? unix(end) : undefined,
    });
    const periods = await waitForChart(chart, { timeoutMs });
    return {
      symbol,
      timeframe,
      infos: chart.infos,
      periods: normalizePeriods(periods, start, end),
    };
  } finally {
    chart.delete();
    await client.end();
  }
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replaceAll('"', '""')}"`;
}

function toCsv(rows, columns) {
  return [
    columns.join(','),
    ...rows.map((row) => columns.map((col) => csvEscape(row[col])).join(',')),
  ].join('\n') + '\n';
}

function writeCsv(file, rows, columns) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, toCsv(rows, columns));
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

function mergeCloseSeries(series) {
  const byDate = new Map();
  for (const [alias, result] of Object.entries(series)) {
    for (const period of result.periods) {
      if (!byDate.has(period.date)) byDate.set(period.date, { date: period.date });
      byDate.get(period.date)[alias] = period.close;
    }
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
}

async function fetchIndicator({
  symbol,
  indicatorId,
  timeframe = 'D',
  range = 300,
  options = {},
  builtIn = false,
  timeoutMs = 45000,
}) {
  const client = new TradingView.Client(authOptions());
  const chart = new client.Session.Chart();

  try {
    chart.setMarket(symbol, { timeframe, range });
    const indicator = builtIn
      ? new TradingView.BuiltInIndicator(indicatorId)
      : await TradingView.getIndicator(
        indicatorId,
        'last',
        process.env.TV_SESSION || '',
        process.env.TV_SIGNATURE || '',
      );
    Object.entries(options).forEach(([key, value]) => indicator.setOption(key, value));
    const study = new chart.Study(indicator);

    const periods = await new Promise((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error(`Indicator timed out after ${timeoutMs}ms`)),
        timeoutMs,
      );
      study.onError((...parts) => {
        clearTimeout(timeout);
        reject(new Error(parts.map(String).join(' ')));
      });
      study.onUpdate(() => {
        clearTimeout(timeout);
        resolve(study.periods);
      });
    });

    return {
      symbol,
      indicator: {
        id: indicatorId,
        description: indicator.description,
        inputs: indicator.inputs,
        plots: indicator.plots,
        builtIn,
      },
      periods,
      graphic: study.graphic,
      strategyReport: study.strategyReport,
    };
  } finally {
    chart.delete();
    await client.end();
  }
}

module.exports = {
  TradingView,
  authOptions,
  requireAuth,
  parseDate,
  unix,
  isoDate,
  approximateRange,
  normalizePeriods,
  waitForChart,
  fetchHistory,
  fetchIndicator,
  toCsv,
  writeCsv,
  writeJson,
  mergeCloseSeries,
};
