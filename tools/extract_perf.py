#!/usr/bin/env python3
"""从测试日志中提取 EagerTime / APTime / Speedup 数据并汇总。

用法:
    python extract_perf.py [日志目录] [-o 输出.csv]

默认扫描 logs/ 目录下的 .txt 日志（仅顶层, 加 -r 可递归子目录），
从形如 "EagerTime: 1.82 ms, APTime: 1.07 ms, Speedup: 1.70x" 的行中提取数据，
测试名和 batch_size 从文件名 test_<name>_bs<N>.txt 解析。
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# 匹配日志中的耗时与加速比
PERF_RE = re.compile(
    r"EagerTime:\s*([\d.]+)\s*ms,\s*APTime:\s*([\d.]+)\s*ms,\s*Speedup:\s*([\d.]+)x"
)
# 从文件名解析测试名与 batch size, 例如 test_matmul_add_gelu_bs32.txt
NAME_RE = re.compile(r"^(.*)_bs(\d+)$")


def parse_file(path: Path):
    """返回 (test_name, batch_size, eager, ap, speedup) 或 None。"""
    text = path.read_text(errors="replace")
    match = None
    for match in PERF_RE.finditer(text):
        pass  # 取最后一次匹配(最终结果)
    if match is None:
        return None

    eager, ap, speedup = (float(x) for x in match.groups())

    stem = path.stem
    name_match = NAME_RE.match(stem)
    if name_match:
        test_name, batch = name_match.group(1), int(name_match.group(2))
    else:
        test_name, batch = stem, None
    return test_name, batch, eager, ap, speedup


def main():
    parser = argparse.ArgumentParser(description="提取日志中的耗时和加速比数据")
    parser.add_argument(
        "log_dir", nargs="?", default="logs", help="日志目录 (默认: logs)"
    )
    parser.add_argument("-o", "--output", help="将结果写入 CSV 文件")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描子目录 (默认只扫描指定目录顶层)",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="仅分析指定的日志文件 (优先于扫描 log_dir), 供批量脚本汇总本次结果",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if args.files:
        txt_files = [Path(f) for f in args.files if Path(f).is_file()]
        if not txt_files:
            print("指定的日志文件均不存在。", file=sys.stderr)
            sys.exit(1)
    elif not log_dir.is_dir():
        print(f"目录不存在: {log_dir}", file=sys.stderr)
        sys.exit(1)
    else:
        txt_files = (
            log_dir.rglob("*.txt") if args.recursive else log_dir.glob("*.txt")
        )

    rows = []
    for txt in sorted(txt_files):
        result = parse_file(txt)
        if result is None:
            continue
        test_name, batch, eager, ap, speedup = result
        try:
            source = str(txt.relative_to(log_dir))
        except ValueError:
            source = str(txt)
        rows.append(
            {
                "source": source,
                "test": test_name,
                "batch_size": batch,
                "eager_ms": eager,
                "ap_ms": ap,
                "speedup": speedup,
            }
        )

    if not rows:
        print("未找到任何耗时数据。", file=sys.stderr)
        sys.exit(1)

    rows.sort(
        key=lambda r: (
            r["test"],
            r["batch_size"] if r["batch_size"] is not None else -1,
            r["source"],
        )
    )
    print_table(rows)

    if args.output:
        with open(args.output, "w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["source", "test", "batch_size", "eager_ms", "ap_ms", "speedup"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已写入 CSV: {args.output}")


def print_table(rows):
    headers = ["Source", "Test", "BS", "Eager(ms)", "AP(ms)", "Speedup"]
    table = [headers]
    for r in rows:
        table.append(
            [
                r["source"],
                r["test"],
                str(r["batch_size"]) if r["batch_size"] is not None else "-",
                f"{r['eager_ms']:.4f}",
                f"{r['ap_ms']:.4f}",
                f"{r['speedup']:.2f}x",
            ]
        )

    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    sep = "-+-".join("-" * w for w in widths)
    for idx, row in enumerate(table):
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if idx == 0:
            print(sep)

    speedups = [r["speedup"] for r in rows]
    print(sep)
    print(
        f"共 {len(rows)} 条, 平均加速比 {sum(speedups) / len(speedups):.2f}x, "
        f"最高 {max(speedups):.2f}x, 最低 {min(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
