"""Launch our own simulator and compare real HTTP execution against offline execution."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from client import Client,HttpTransport
from simulator import LocalSimulator,make_case
from strategy import Policy

def main():
    out=Path('results/http_smoke'); out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    server=subprocess.Popen([sys.executable,'simulator.py','--seed','42','--port',str(port)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        line=server.stdout.readline()
        if not line.startswith('SELF SIMULATOR ONLY'): raise RuntimeError(server.stderr.read())
        client=Client(HttpTransport(f'http://127.0.0.1:{port}'),log_path=out/'actions.jsonl')
        try: result=Policy(client).run()
        finally: client.close()
        sim=LocalSimulator(make_case(42),42)
        baseline=Policy(Client(sim)).run()
        assert result['completed'] and baseline['completed']
        assert result['cleared_channels']==baseline['cleared_channels']
        assert abs(result['virtual_time_s']-baseline['virtual_time_s'])<1e-5
        record={'test_kind':'SELF_BUILT_LOOPBACK_HTTP','matched_offline':True,'http_runtime_s':result['runtime_s'],
                'http_virtual_time_s':result['virtual_time_s'],'offline_virtual_time_s':baseline['virtual_time_s'],
                'cleared_channels':result['cleared_channels'],'http_requests':client.counter}
        (out/'verification.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record,indent=2))
    finally:
        server.terminate()
        try: server.wait(timeout=5)
        except subprocess.TimeoutExpired: server.kill(); server.wait()

if __name__=='__main__': main()
