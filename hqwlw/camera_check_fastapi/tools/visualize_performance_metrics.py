#!/usr/bin/env python3
"""Generate an HTML dashboard for the performance metrics endpoint.

The FastAPI service in :mod:`camera_check_fastapi` exposes lightweight
performance statistics at ``/metrics/performance`` and
``/metrics/performance/{stream_name}``.  The payload is convenient for machines
but not for humans – this script converts the JSON structure into an
ECharts-powered HTML page so operators can inspect the timings without setting
up a separate monitoring stack.

Examples
--------
Fetch metrics from a running service and open the generated report::

    python visualize_performance_metrics.py \
        http://localhost:8000/metrics/performance \
        --output performance.html
    python -m webbrowser performance.html

The script also accepts offline snapshots (either a file path or ``-`` for
``stdin``)::

    curl http://localhost:8000/metrics/performance > snapshot.json
    python visualize_performance_metrics.py snapshot.json

The resulting ``performance_metrics.html`` can be emailed or attached to bug
reports – the charts are rendered entirely in the browser, no external service
is required beyond loading the ECharts runtime from a CDN.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import urlopen

DEFAULT_ENDPOINT = "http://localhost:8000/metrics/performance"

HTML_TEMPLATE = textwrap.dedent(
    """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8" />
        <title>__TITLE__</title>
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js" integrity="sha384-ceOsJAlBnZ2sUmSwE1s9hy9GPzfThl/8wsPjO1qKpc0W8E2ZvmOef39/CWKiG1lf" crossorigin="anonymous"></script>
        <style>
            body {
                font-family: "Segoe UI", "PingFang SC", "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                background: #f5f6fa;
                color: #222;
            }
            header {
                background: #1f6feb;
                color: white;
                padding: 1.5rem 2rem;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
            }
            header h1 {
                margin: 0 0 0.5rem;
                font-size: 1.8rem;
            }
            header p {
                margin: 0.25rem 0;
                line-height: 1.4;
            }
            main {
                padding: 1.5rem 2rem 3rem;
                max-width: 1080px;
                margin: 0 auto;
            }
            section {
                margin-bottom: 2.5rem;
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(31, 35, 41, 0.15);
                padding: 1.5rem;
            }
            section h2 {
                margin-top: 0;
            }
            .charts-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 1rem;
            }
            .chart-card {
                background: #fdfdfd;
                border-radius: 10px;
                border: 1px solid #e4e6eb;
                padding: 0.75rem;
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            .chart-card h3 {
                margin: 0;
                font-size: 1.05rem;
            }
            .summary-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.85rem;
            }
            .summary-table th,
            .summary-table td {
                text-align: left;
                padding: 0.1rem 0.3rem;
                border-bottom: 1px dotted #d0d7de;
            }
            .chart {
                width: 100%;
                height: 280px;
            }
            footer {
                text-align: center;
                padding: 1.5rem;
                color: #6e7781;
                font-size: 0.85rem;
            }
            .muted {
                color: #6e7781;
            }
            .stream-block + .stream-block {
                margin-top: 1.5rem;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>__TITLE__</h1>
            <p>数据更新时间：<span id="updated-at">未知</span></p>
            <p class="muted">报告生成时间（UTC）：__GENERATED_AT__</p>
        </header>
        <main>
            <section id="global-section" hidden>
                <h2>全局阶段耗时</h2>
                <div id="global-charts" class="charts-grid"></div>
            </section>
            <section id="streams-section" hidden>
                <h2>视频流阶段耗时</h2>
                <div id="streams-container"></div>
            </section>
            <section id="frames-section" hidden>
                <h2>帧级时间线</h2>
                <div id="frames-container"></div>
            </section>
        </main>
        <footer>
            由 <code>visualize_performance_metrics.py</code> 生成 – 拖拽或点击图例可交互查看数据。
        </footer>
        <script>
            const performanceData = __PAYLOAD__;

            function formatNumber(value) {
                if (value === null || value === undefined) {
                    return '—';
                }
                const numberValue = Number(value);
                if (!Number.isFinite(numberValue)) {
                    return '—';
                }
                return numberValue.toFixed(3);
            }

            function fillSummary(table, stats) {
                const rows = [
                    ['样本数', stats.count ?? '—'],
                    ['平均值 (ms)', formatNumber(stats.avg)],
                    ['95 分位 (ms)', formatNumber(stats.p95)],
                    ['最大值 (ms)', formatNumber(stats.max)],
                    ['最小值 (ms)', formatNumber(stats.min)],
                    ['最新值 (ms)', formatNumber(stats.last)],
                ];
                rows.forEach(([label, value]) => {
                    const tr = document.createElement('tr');
                    const th = document.createElement('th');
                    th.textContent = label;
                    const td = document.createElement('td');
                    td.textContent = value;
                    tr.appendChild(th);
                    tr.appendChild(td);
                    table.appendChild(tr);
                });
            }

            function renderStageCharts(sectionElement, containerElement, statsMap, titlePrefix = '') {
                const entries = Object.entries(statsMap || {});
                if (!entries.length) {
                    return;
                }
                sectionElement.hidden = false;
                entries.forEach(([stageName, stats]) => {
                    const card = document.createElement('div');
                    card.className = 'chart-card';

                    const heading = document.createElement('h3');
                    heading.textContent = titlePrefix + stageName;
                    card.appendChild(heading);

                    const table = document.createElement('table');
                    table.className = 'summary-table';
                    fillSummary(table, stats);
                    card.appendChild(table);

                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'chart';
                    card.appendChild(chartDiv);
                    containerElement.appendChild(card);

                    const history = (stats.history || []).map(item => (item === null ? null : Number(item)));
                    if (!history.length) {
                        chartDiv.textContent = '暂无历史数据';
                        chartDiv.style.display = 'flex';
                        chartDiv.style.alignItems = 'center';
                        chartDiv.style.justifyContent = 'center';
                        return;
                    }
                    const xData = history.map((_, index) => index + 1);
                    const chart = echarts.init(chartDiv);
                    chart.setOption({
                        tooltip: { trigger: 'axis' },
                        xAxis: {
                            type: 'category',
                            data: xData,
                            boundaryGap: false,
                            name: '最近样本序号'
                        },
                        yAxis: { type: 'value', name: '耗时 (ms)' },
                        series: [{
                            name: stageName,
                            type: 'line',
                            smooth: true,
                            showSymbol: false,
                            data: history,
                            areaStyle: { opacity: 0.1 }
                        }],
                    });
                });
            }

            function renderStreams(streams) {
                const section = document.getElementById('streams-section');
                const container = document.getElementById('streams-container');
                const entries = Object.entries(streams || {});
                if (!entries.length) {
                    return;
                }
                section.hidden = false;
                entries.forEach(([streamName, statsMap]) => {
                    const wrapper = document.createElement('div');
                    wrapper.className = 'stream-block';
                    const heading = document.createElement('h3');
                    heading.textContent = '流：' + streamName;
                    wrapper.appendChild(heading);

                    const grid = document.createElement('div');
                    grid.className = 'charts-grid';
                    wrapper.appendChild(grid);
                    container.appendChild(wrapper);

                    renderStageCharts(wrapper, grid, statsMap, '阶段：');
                });
            }

            function renderFrames(frames, streamName) {
                if (!frames || !frames.length) {
                    return;
                }
                const section = document.getElementById('frames-section');
                const container = document.getElementById('frames-container');
                section.hidden = false;

                const stageSet = new Set();
                frames.forEach(frame => {
                    Object.keys(frame.timings || {}).forEach(name => stageSet.add(name));
                });
                const stageNames = Array.from(stageSet);

                const chartDiv = document.createElement('div');
                chartDiv.className = 'chart';
                container.appendChild(chartDiv);

                const xData = frames.map(frame => frame.recorded_at || '');
                const series = stageNames.map(name => ({
                    name: name,
                    type: 'line',
                    smooth: true,
                    connectNulls: true,
                    data: frames.map(frame => {
                        const value = (frame.timings || {})[name];
                        return value === undefined || value === null ? null : Number(value);
                    }),
                }));

                const chart = echarts.init(chartDiv);
                chart.setOption({
                    title: streamName ? { text: streamName + ' 的帧耗时' } : undefined,
                    tooltip: { trigger: 'axis' },
                    legend: { data: stageNames },
                    xAxis: {
                        type: 'category',
                        data: xData,
                        boundaryGap: false,
                        axisLabel: { rotate: 45 },
                    },
                    yAxis: { type: 'value', name: '耗时 (ms)' },
                    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
                    series: series,
                });
            }

            function main() {
                if (!performanceData) {
                    return;
                }
                const updated = performanceData.updated_at;
                if (updated) {
                    document.getElementById('updated-at').textContent = updated;
                } else if (performanceData.frames && performanceData.frames.length) {
                    const lastFrame = performanceData.frames[performanceData.frames.length - 1];
                    if (lastFrame && lastFrame.recorded_at) {
                        document.getElementById('updated-at').textContent = lastFrame.recorded_at;
                    }
                }

                const globalSection = document.getElementById('global-section');
                const globalContainer = document.getElementById('global-charts');
                renderStageCharts(globalSection, globalContainer, performanceData.global || {});
                renderStreams(performanceData.streams || {});
                renderFrames(performanceData.frames || [], performanceData.stream || null);
            }

            main();
        </script>
    </body>
    </html>
    """
)



def _load_payload(source: str) -> Dict[str, Any]:
    """Load the JSON payload from *source*.

    *source* may be ``-`` (meaning ``stdin``), an HTTP(S) URL, or a local file
    path.  The function returns the decoded JSON document.
    """

    if source == "-":
        return json.load(sys.stdin)

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        with urlopen(source) as response:  # type: ignore[arg-type]
            charset = response.headers.get_content_charset() or "utf-8"
            data = response.read().decode(charset)
            return json.loads(data)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Unable to locate metrics snapshot: {source}")
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_stream(data: Dict[str, Any], stream: str) -> Dict[str, Any]:
    """Extract metrics for a specific *stream* from an aggregate snapshot."""

    if not stream:
        return data

    if data.get("stream") == stream:
        return data

    streams = data.get("streams", {}) or {}
    stream_stats = streams.get(stream)
    if stream_stats is None:
        available = ", ".join(sorted(streams)) or "<none>"
        raise SystemExit(
            f"Stream '{stream}' not found in snapshot. Available streams: {available}"
        )

    filtered = dict(data)
    filtered["stream"] = stream
    filtered["stages"] = stream_stats
    filtered["streams"] = {stream: stream_stats}
    filtered.setdefault("frames", [])
    return filtered


def _normalise_numeric_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Decimal-like objects to floats so they can be serialised safely."""

    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        if isinstance(obj, (int, float)) or obj is None:
            return obj
        try:
            return float(obj)
        except Exception:
            return obj

    return convert(data)


def _build_html(data: Dict[str, Any], title: str) -> str:
    """Return the HTML document string embedding *data* as charts."""

    safe_payload = _normalise_numeric_fields(data)
    payload_json = json.dumps(safe_payload, ensure_ascii=False, indent=2)
    payload_json = payload_json.replace("</script>", "<\\/script>")

    generated_at = datetime.now(timezone.utc).isoformat()
    document = HTML_TEMPLATE
    document = document.replace("__TITLE__", html.escape(title))
    document = document.replace("__GENERATED_AT__", html.escape(generated_at))
    document = document.replace("__PAYLOAD__", payload_json)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将性能监控接口返回的数据转换为交互式 HTML 仪表盘",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_ENDPOINT,
        help="数据来源，可以是 HTTP(S) URL、本地 JSON 文件路径，或 '-' 代表标准输入",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="performance_metrics.html",
        help="输出的 HTML 文件路径",
    )
    parser.add_argument(
        "--stream",
        help="只展示指定视频流的数据（聚合接口下有效）",
    )
    parser.add_argument(
        "--title",
        default="性能检测数据可视化",
        help="生成页面的标题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_payload(args.source)
    if args.stream:
        data = _filter_stream(data, args.stream)
    html_document = _build_html(data, args.title)

    output_path = Path(args.output)
    output_path.write_text(html_document, encoding="utf-8")
    print(f"Saved dashboard to {output_path.resolve()}")


if __name__ == "__main__":
    main()
