#!/usr/bin/env python3
"""CASBOT02 硬限位零偏校准 Web 控制台。

启动后在浏览器中打开 http://<host>:8088 即可使用。

Usage::

    ros2 run zero_offset_calibration calibration_web_ui
    ros2 run zero_offset_calibration calibration_web_ui --port 9000
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .hard_stop_calibration import write_joint_pos_offset_yaml

_DEFAULT_PORT = 8088

_OFFSET_PATHS = [
    Path("src/config/joint_pos_offset.yaml"),
    Path("/workspace/hl_motion/hl_config/joint_pos_offset.yaml"),
]
_MAX_LOG_LINES = 10_000
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Calibration subprocess manager
# ---------------------------------------------------------------------------

class CalibrationProcess:
    """管理标定子进程的生命周期：启动 / 取消 / 日志收集。"""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._logs: List[str] = []
        self._lock = threading.RLock()
        self._status = "idle"
        self._return_code: Optional[int] = None
        self._command = ""

    # -- public API --

    def start(
        self,
        arm: str,
        instrument: str,
        persist: bool = True,
        skip_on_timeout: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._status == "running":
                return {"ok": False, "error": "标定正在运行中"}

            cmd: List[str] = [
                "ros2", "run", "zero_offset_calibration",
                "ros2_upper_body_hardware",
                "--arm", arm,
                "--instrument", instrument,
            ]
            if persist:
                cmd.append("--persist")
            if skip_on_timeout:
                cmd.append("--skip-on-timeout")

            self._command = " ".join(cmd)
            self._logs.clear()
            self._return_code = None
            self._status = "running"
            self._logs.append(f"$ {self._command}")

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )
            except Exception as exc:
                self._status = "error"
                self._logs.append(f"启动失败: {exc}")
                return {"ok": False, "error": str(exc)}

            t = threading.Thread(target=self._read_output, daemon=True)
            t.start()
            return {"ok": True}

    def cancel(self) -> Dict[str, Any]:
        proc: Optional[subprocess.Popen]
        with self._lock:
            proc = self._proc
            running = self._status == "running"

        if proc is None or not running:
            self._disable_debug()
            return {"ok": True, "message": "未在运行，已尝试关闭调试模式"}

        self._append("[Web] 正在取消标定...")

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass

        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            self._append("[Web] 进程未响应 SIGINT，强制终止...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass

        with self._lock:
            self._status = "cancelled"

        self._disable_debug()
        self._append("[Web] 标定已取消")
        return {"ok": True}

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "return_code": self._return_code,
                "log_count": len(self._logs),
                "command": self._command,
            }

    def get_logs(self, since: int = 0) -> Dict[str, Any]:
        with self._lock:
            total = len(self._logs)
            lines = self._logs[since:] if since < total else []
            return {"lines": lines, "total": total}

    # -- internal --

    def _append(self, line: str) -> None:
        with self._lock:
            if len(self._logs) < _MAX_LOG_LINES:
                self._logs.append(line)

    def _read_output(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                self._append(_strip_ansi(raw.rstrip("\n")))
        except Exception:
            pass
        finally:
            rc = proc.wait()
            with self._lock:
                self._return_code = rc
                if self._status == "running":
                    self._status = "done" if rc == 0 else "error"
                    if rc != 0:
                        self._append(f"进程退出，返回码: {rc}")

    def apply_offset(self) -> Dict[str, Any]:
        """调用 /motion/set_joint_offset 服务通知 hlmotion 重新加载零偏。"""
        self._append("[Web] 正在请求 /motion/set_joint_offset ...")
        try:
            r = subprocess.run(
                [
                    "ros2", "service", "call",
                    "/motion/set_joint_offset",
                    "std_srvs/srv/SetBool",
                    "{data: true}",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                self._append("[Web] /motion/set_joint_offset 调用成功")
                return {"ok": True, "message": "零偏已应用"}
            else:
                msg = (r.stderr or r.stdout or "").strip()
                self._append(f"[Web] /motion/set_joint_offset 返回码 {r.returncode}: {msg}")
                return {"ok": False, "error": f"返回码 {r.returncode}"}
        except subprocess.TimeoutExpired:
            self._append("[Web] /motion/set_joint_offset 超时（服务可能不可用）")
            return {"ok": False, "error": "服务调用超时"}
        except Exception as exc:
            self._append(f"[Web] /motion/set_joint_offset 失败: {exc}")
            return {"ok": False, "error": str(exc)}

    def reset_offsets(self) -> Dict[str, Any]:
        """将所有零偏 YAML 文件清零。"""
        self._append("[Web] 正在清零零偏数据...")
        empty: Dict[str, float] = {}
        header = ("CASBOT02 upper body — hard-stop zero offsets (radians).",)
        written: List[str] = []
        errors: List[str] = []
        for p in _OFFSET_PATHS:
            try:
                write_joint_pos_offset_yaml(p, empty, "both", header_lines=header)
                written.append(str(p))
                self._append(f"[Web] 已清零: {p}")
            except OSError as exc:
                errors.append(f"{p}: {exc}")
                self._append(f"[Web] 清零失败: {p} — {exc}")
        if errors:
            return {"ok": False, "error": "; ".join(errors), "written": written}
        return {"ok": True, "message": f"已清零 {len(written)} 个文件"}

    def _disable_debug(self) -> None:
        self._append("[Web] 正在关闭上半身调试模式...")
        try:
            r = subprocess.run(
                [
                    "ros2", "service", "call",
                    "/motion/upper_body_debug",
                    "std_srvs/srv/SetBool",
                    "{data: false}",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                self._append("[Web] 上半身调试模式已关闭")
            else:
                self._append(f"[Web] 关闭调试模式返回码 {r.returncode}")
        except subprocess.TimeoutExpired:
            self._append("[Web] 关闭调试模式超时（服务可能不可用）")
        except Exception as exc:
            self._append(f"[Web] 关闭调试模式失败: {exc}")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_manager = CalibrationProcess()


class _Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # noqa: ARG002
        pass

    # -- routing --

    def do_GET(self) -> None:
        if self.path == "/":
            self._html(_HTML_PAGE)
        elif self.path == "/api/state":
            self._json(_manager.get_state())
        elif self.path.startswith("/api/logs"):
            since = 0
            if "?" in self.path:
                for part in self.path.split("?", 1)[1].split("&"):
                    if part.startswith("since="):
                        try:
                            since = int(part.split("=", 1)[1])
                        except ValueError:
                            pass
            self._json(_manager.get_logs(since))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/start":
            body = self._body()
            result = _manager.start(
                arm=body.get("arm", "right"),
                instrument=body.get("instrument", "bass"),
                persist=bool(body.get("persist", True)),
                skip_on_timeout=bool(body.get("skip_on_timeout", True)),
            )
            self._json(result)
        elif self.path == "/api/cancel":
            threading.Thread(target=_manager.cancel, daemon=True).start()
            self._json({"ok": True, "message": "取消请求已提交"})
        elif self.path == "/api/apply_offset":
            result = _manager.apply_offset()
            self._json(result)
        elif self.path == "/api/reset_offsets":
            result = _manager.reset_offsets()
            self._json(result)
        else:
            self.send_error(404)

    # -- helpers --

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        try:
            return json.loads(self.rfile.read(n))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _json(self, data: dict, code: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _html(self, text: str) -> None:
        raw = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


# ---------------------------------------------------------------------------
# Embedded HTML / CSS / JS
# ---------------------------------------------------------------------------

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CASBOT02 硬限位零偏校准</title>
<style>
:root{
  --bg:#0f172a;--card:#1e293b;--input:#0f172a;--border:#334155;
  --text:#f1f5f9;--dim:#94a3b8;--accent:#3b82f6;--accent2:#2563eb;
  --ok:#22c55e;--err:#ef4444;--warn:#f59e0b;--term:#020617;
  --mono:'Consolas','Monaco','Courier New',monospace;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  --r:8px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--sans);background:var(--bg);color:var(--text);min-height:100vh}

.header{background:var(--card);border-bottom:1px solid var(--border);padding:14px 24px;
  display:flex;align-items:center;gap:12px}
.header h1{font-size:18px;font-weight:600;letter-spacing:.02em}
.header .dot{width:8px;height:8px;border-radius:50%;background:var(--ok)}

.main{max-width:1200px;margin:20px auto;padding:0 20px;
  display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}
@media(max-width:800px){.main{grid-template-columns:1fr}}

.panel{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
.panel h2{font-size:13px;font-weight:600;color:var(--dim);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:16px}

.fg{margin-bottom:16px}
.fg>label{display:block;font-size:13px;font-weight:500;color:var(--dim);margin-bottom:6px}

.radio-row{display:flex;border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
.radio-row input{display:none}
.radio-row label{flex:1;padding:9px 0;text-align:center;font-size:14px;cursor:pointer;
  background:var(--input);color:var(--dim);transition:all .15s;user-select:none}
.radio-row label:not(:last-of-type){border-right:1px solid var(--border)}
.radio-row input:checked+label{background:var(--accent);color:#fff;font-weight:600}

select{width:100%;padding:9px 32px 9px 12px;background:var(--input);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text);font-size:14px;cursor:pointer;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12'%3E%3Cpath fill='%2394a3b8' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 10px center}
select:focus{outline:2px solid var(--accent);outline-offset:-1px}

.cb{display:flex;align-items:center;gap:8px;margin-bottom:10px;font-size:14px;cursor:pointer}
.cb input{width:16px;height:16px;accent-color:var(--accent);cursor:pointer}

.cmd{margin-top:12px;padding:8px 10px;background:var(--term);border:1px solid var(--border);
  border-radius:var(--r);font:12px var(--mono);color:var(--dim);word-break:break-all;
  line-height:1.5;min-height:36px}

.btns{display:flex;gap:10px;margin-top:16px}
.btn{flex:1;padding:11px 0;border:none;border-radius:var(--r);font-size:14px;font-weight:600;
  cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center;gap:6px}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-go{background:var(--accent);color:#fff}
.btn-go:hover:not(:disabled){background:var(--accent2)}
.btn-stop{background:var(--err);color:#fff}
.btn-stop:hover:not(:disabled){background:#dc2626}
.btn-apply{background:#8b5cf6;color:#fff}
.btn-apply:hover:not(:disabled){background:#7c3aed}
.btn-reset{background:#64748b;color:#fff}
.btn-reset:hover:not(:disabled){background:#475569}

.status{display:flex;align-items:center;gap:8px;margin-top:14px;padding:10px 12px;
  background:var(--input);border:1px solid var(--border);border-radius:var(--r);font-size:13px}
.sd{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.sd.idle{background:var(--dim)}
.sd.running{background:var(--ok);animation:pulse 1.4s infinite}
.sd.done{background:var(--ok)}
.sd.error{background:var(--err)}
.sd.cancelled{background:var(--warn)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

.log-panel{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  display:flex;flex-direction:column;box-shadow:0 2px 8px rgba(0,0,0,.25);
  overflow:hidden;height:var(--panel-h,480px)}
.log-head{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;
  border-bottom:1px solid var(--border)}
.log-head h2{font-size:13px;font-weight:600;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.log-head button{background:none;border:1px solid var(--border);border-radius:var(--r);
  color:var(--dim);padding:4px 10px;font-size:12px;cursor:pointer}
.log-head button:hover{color:var(--text);border-color:var(--text)}

.log-body{flex:1;background:var(--term);padding:12px 14px;overflow-y:auto;
  font:13px/1.7 var(--mono);min-height:0}
.ll{white-space:pre-wrap;word-break:break-all}
.ll.i{color:#e2e8f0}
.ll.w{color:#fbbf24}
.ll.e{color:#f87171}
.ll.s{color:#60a5fa;font-style:italic}
</style>
</head>
<body>

<div class="header">
  <div class="dot"></div>
  <h1>CASBOT02 硬限位零偏校准控制台</h1>
</div>

<div class="main">
  <!-- left: controls -->
  <div>
    <div class="panel">
      <h2>标定配置</h2>

      <div class="fg">
        <label>手臂选择</label>
        <div class="radio-row">
          <input type="radio" name="arm" id="arm-l" value="left" class="fc">
          <label for="arm-l">左臂</label>
          <input type="radio" name="arm" id="arm-r" value="right" class="fc" checked>
          <label for="arm-r">右臂</label>
        </div>
      </div>

      <div class="fg">
        <label for="inst">乐器选择</label>
        <select id="inst" class="fc">
          <option value="bass">Bass (贝斯)</option>
          <option value="guitar" selected>Guitar (吉他)</option>
          <option value="auto">Auto (自动识别)</option>
          <option value="none">None (裸机)</option>
          <option value="keyboard">Keyboard (电子琴)</option>
        </select>
      </div>

      <label class="cb"><input type="checkbox" id="persist" class="fc" checked>持久化写入 YAML (--persist)</label>
      <label class="cb"><input type="checkbox" id="skip" class="fc" checked>超时跳过继续 (--skip-on-timeout)</label>

      <div class="cmd" id="cmd-preview"></div>

      <div class="btns">
        <button class="btn btn-go" id="btn-go" onclick="doStart()">&#9654; 开始标定</button>
        <button class="btn btn-stop" id="btn-stop" onclick="doCancel()" disabled>&#10005; 取消</button>
      </div>

      <div class="btns" style="margin-top:8px">
        <button class="btn btn-apply" id="btn-apply" onclick="doApplyOffset()">&#8635; 应用零偏</button>
        <button class="btn btn-reset" id="btn-reset" onclick="doResetOffsets()">&#10060; 清零</button>
      </div>

      <div class="status">
        <div class="sd idle" id="sd"></div>
        <span id="st">空闲</span>
      </div>
    </div>
  </div>

  <!-- right: logs -->
  <div class="log-panel" id="log-panel">
    <div class="log-head">
      <h2>实时日志</h2>
      <button onclick="clearLogs()">清空</button>
    </div>
    <div class="log-body" id="log"></div>
  </div>
</div>

<script>
const $=s=>document.getElementById(s);
let offset=0, polling=null, lastSt='idle';

function buildCmd(){
  const arm=document.querySelector('input[name=arm]:checked').value;
  const inst=$('inst').value;
  const p=$('persist').checked;
  const sk=$('skip').checked;
  let c='ros2 run zero_offset_calibration ros2_upper_body_hardware --arm '+arm+' --instrument '+inst;
  if(p) c+=' --persist';
  if(sk) c+=' --skip-on-timeout';
  return c;
}
function updPreview(){$('cmd-preview').textContent='$ '+buildCmd()}

function cls(line){
  if(line.startsWith('[Web')) return 's';
  if(line.includes('[ERROR]')) return 'e';
  if(line.includes('[WARN]')) return 'w';
  if(line.includes('[INFO]')) return 'i';
  return 'i';
}

function addLines(lines){
  const el=$('log');
  const atBot=el.scrollHeight-el.scrollTop-el.clientHeight<40;
  for(const l of lines){
    const d=document.createElement('div');
    d.className='ll '+cls(l);
    d.textContent=l;
    el.appendChild(d);
  }
  if(atBot) el.scrollTop=el.scrollHeight;
}

function clearLogs(){$('log').innerHTML='';offset=0}

const stMap={idle:'空闲',running:'运行中',done:'完成',error:'异常退出',cancelled:'已取消'};

function updUI(s){
  $('sd').className='sd '+s.status;
  $('st').textContent=stMap[s.status]||s.status;
  $('btn-go').disabled=s.status==='running';
  $('btn-stop').disabled=s.status!=='running';
  $('btn-apply').disabled=s.status==='running';
  $('btn-reset').disabled=s.status==='running';
  document.querySelectorAll('.fc').forEach(e=>e.disabled=s.status==='running');
}

async function poll(){
  try{
    const s=await fetch('/api/state').then(r=>r.json());
    updUI(s);
    lastSt=s.status;
    if(s.log_count>offset){
      const d=await fetch('/api/logs?since='+offset).then(r=>r.json());
      if(d.lines.length) addLines(d.lines);
      offset=d.total;
    }
  }catch(e){console.error(e)}
  polling=setTimeout(poll, lastSt==='running'?300:1000);
}

async function doStart(){
  const arm=document.querySelector('input[name=arm]:checked').value;
  const inst=$('inst').value;
  clearLogs();
  const r=await fetch('/api/start',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({arm:arm,instrument:inst,
      persist:$('persist').checked,skip_on_timeout:$('skip').checked})
  }).then(r=>r.json());
  if(!r.ok) alert(r.error||'启动失败');
}

async function doCancel(){
  if(!confirm('确定要取消标定并关闭上半身调试模式？')) return;
  await fetch('/api/cancel',{method:'POST'});
}

async function doResetOffsets(){
  if(!confirm('确定要将所有零偏数据清零？')) return;
  const b=$('btn-reset');
  b.disabled=true;
  b.textContent='清零中...';
  try{
    const r=await fetch('/api/reset_offsets',{method:'POST'}).then(r=>r.json());
    if(!r.ok) alert(r.error||'清零失败');
  }catch(e){alert('请求失败: '+e)}
  finally{b.disabled=false;b.textContent='\u274C 清零'}
}

async function doApplyOffset(){
  const b=$('btn-apply');
  b.disabled=true;
  b.textContent='请求中...';
  try{
    const r=await fetch('/api/apply_offset',{method:'POST'}).then(r=>r.json());
    if(!r.ok) alert(r.error||'应用失败');
  }catch(e){alert('请求失败: '+e)}
  finally{b.disabled=false;b.textContent='\u21BB 应用零偏'}
}

function syncHeight(){
  const p=document.querySelector('.panel');
  if(p) document.querySelector('.log-panel').style.setProperty('--panel-h',p.offsetHeight+'px');
}
window.addEventListener('resize',syncHeight);
document.querySelectorAll('.fc').forEach(e=>e.addEventListener('change',updPreview));
updPreview();
syncHeight();
poll();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CASBOT02 硬限位零偏校准 Web 控制台")
    ap.add_argument("--port", type=int, default=_DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    server = http.server.ThreadingHTTPServer((args.host, args.port), _Handler)
    if args.host in ("0.0.0.0", ""):
        import socket as _socket
        _ip = "0.0.0.0"
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            _ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        print(f"CASBOT02 硬限位零偏校准 Web 控制台已启动:")
        print(f"  本机访问: http://localhost:{args.port}")
        print(f"  局域网访问: http://{_ip}:{args.port}")
    else:
        print(f"CASBOT02 硬限位零偏校准 Web 控制台已启动: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭...")
        if _manager.get_state()["status"] == "running":
            print("正在取消运行中的标定...")
            _manager.cancel()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
