#!/usr/bin/env python3
"""TCP discard sink for the Pico 2 W link-ceiling bench (Stage 1).

Receives what ``firmware/nodes/bench_cyw43_iperf`` blasts and reports
receiver-side goodput per connection, so the Pico's ACK-counted number has an
independent cross-check. Standard library only: it must run on the ostree
Bluefin box without touching the minimappr env.

One JSON line per finished connection goes to stdout (and ``--log`` if given):
``{"peer":..., "seconds":..., "bytes":..., "mbps":..., "per_second_mbps":[...]}``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

RECEIVE_CHUNK_BYTES = 65536


async def handle_connection(reader, writer, log_path: str | None) -> None:
    peer = writer.get_extra_info("peername")
    started = time.monotonic()
    second_start = started
    second_bytes = 0
    total_bytes = 0
    per_second_mbps: list[float] = []
    while True:
        data = await reader.read(RECEIVE_CHUNK_BYTES)
        if not data:
            break
        now = time.monotonic()
        total_bytes += len(data)
        second_bytes += len(data)
        while now - second_start >= 1.0:
            per_second_mbps.append(round(second_bytes * 8 / 1e6, 3))
            second_bytes = 0
            second_start += 1.0
    elapsed = max(time.monotonic() - started, 1e-9)
    writer.close()
    summary = {
        "peer": f"{peer[0]}:{peer[1]}" if peer else None,
        "seconds": round(elapsed, 3),
        "bytes": total_bytes,
        "mbps": round(total_bytes * 8 / elapsed / 1e6, 3),
        "per_second_mbps": per_second_mbps,
        "finished_at": time.time(),
    }
    line = json.dumps(summary)
    print(line, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5201)
    parser.add_argument("--log", default=None, help="append per-connection JSON lines here")
    args = parser.parse_args()
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, args.log), args.host, args.port
    )
    print(json.dumps({"listening": f"{args.host}:{args.port}"}), flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
