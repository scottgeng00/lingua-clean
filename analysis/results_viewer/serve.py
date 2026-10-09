"""Serve the results viewer, or export it as one self-contained HTML file.

    # live (re-collects on page load / Refresh, at most every --min-interval seconds)
    .venv/bin/python analysis/results_viewer/serve.py --port 8765
    # then from your laptop:  ssh -L 8765:localhost:8765 <login-node>  ->  http://localhost:8765

    # static snapshot with the data inlined (open anywhere, no server)
    .venv/bin/python analysis/results_viewer/serve.py --export /tmp/results.html
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class DataCache:
    def __init__(self, registry, min_interval):
        self.registry, self.min_interval = registry, min_interval
        self.lock = threading.Lock()
        self.blob, self.at = None, 0.0

    def get(self, force=False):
        with self.lock:
            if self.blob is None or (force and time.time() - self.at >= self.min_interval):
                self.blob = collect.to_json(collect.collect(self.registry)).encode()
                self.at = time.time()
            return self.blob


def page_with_inline_data(data_json):
    with open(os.path.join(STATIC, "index.html")) as f:
        html = f.read()
    css = open(os.path.join(STATIC, "style.css")).read()
    js = open(os.path.join(STATIC, "app.js")).read()
    # "</" would end the <script> element early
    data_json = data_json.replace("</", "<\\/")
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")
    html = html.replace(
        '<script src="app.js"></script>',
        f"<script>window.RESULTS_DATA = {data_json};</script>\n<script>\n{js}\n</script>",
    )
    return html


def make_handler(cache):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=STATIC, **kw)

        def do_GET(self):
            if self.path.startswith("/api/data"):
                try:
                    body = cache.get(force="refresh=1" in self.path)
                except Exception as e:  # report collection errors to the page
                    body, code = json.dumps({"error": repr(e)}).encode(), 500
                else:
                    code = 200
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()

        def log_message(self, fmt, *args):
            if "/api/data" in (args[0] if args else ""):
                sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--registry", default=None, help="runs.yaml (default: next to this file)")
    ap.add_argument("--min-interval", type=float, default=20.0, help="min seconds between re-collections")
    ap.add_argument("--export", metavar="HTML", help="write a self-contained snapshot and exit")
    args = ap.parse_args()

    if args.export:
        data = collect.to_json(collect.collect(args.registry))
        with open(args.export, "w") as f:
            f.write(page_with_inline_data(data))
        print(f"wrote {args.export} ({os.path.getsize(args.export) / 1e6:.2f} MB)")
        return

    cache = DataCache(args.registry, args.min_interval)
    cache.get()  # collect once up front so the first page load is fast
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((args.host, args.port), make_handler(cache)) as srv:
        print(f"results viewer on http://{args.host}:{args.port}  "
              f"(from a laptop: ssh -L {args.port}:localhost:{args.port} {os.uname().nodename})")
        srv.serve_forever()


if __name__ == "__main__":
    main()
