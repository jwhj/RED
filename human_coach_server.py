from typing import Dict, List, Tuple
import os
import json
import csv
import random
import time
import uuid
import pickle
import subprocess
import socket
import lmdb
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel


class UserConfig(BaseModel):
    id_to_model_type: List[str]
    match_seeds: List[List[int]]


users_config: Dict[str, UserConfig] = {}
with open('users_config.json', 'r') as f:
    tmp = json.load(f)
    for username in tmp:
        users_config[username] = UserConfig(**tmp[username])

with open('frozen_id.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        del users_config[row[0]]

env = lmdb.open('/workspace/human_eval_results')

model_paths = {
    'red': 'ckpt/red-1.pt',
    'irl': 'ckpt/airl-1.pt',
    'rl': 'ckpt/rl-1.pt',
    'joint': 'ckpt/joint-1.pt',
    'switch': 'ckpt/rl-1.pt',
    'bc': 'ckpt/executor_rnn_with_cont_bsz128/best_checkpoint.pt'
    # 'best': '/mnt/shared/best',
    # 'irl': '/mnt/shared/airl-disc-full',
    # 'rl': '/mnt/shared/rl-latest',
    # 'switch': '/mnt/shared/rl-latest',
    # 'rl-joint': '/mnt/shared/rl-joint',
    # 'bc': '/mnt/shared/minirts/scripts/behavior_clone/saved_models/executor_rnn_with_cont_bsz128/best_checkpoint.pt',
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor()
running_games: Dict[str, subprocess.Popen] = {}
timestamps: Dict[str, float] = {}
cnt = 0


def get_available_port():
    return 8002
    sock = socket.socket()
    sock.bind(('', 0))
    _, port = sock.getsockname()
    sock.close()
    return port


def check_heartbeat(port: int, interval: float):
    while True:
        time.sleep(interval)
        t = time.time()
        if port not in timestamps or t - timestamps[port] >= interval:
            print(f'terminated: {port}')
            p = running_games[port]
            p.terminate()
            del running_games[port]
            del timestamps[port]
            break
        else:
            print(f'ok: {port}')


def start_game(
    port: int,
    seed: int,
    model_type: str,
    save_dir: str = None,
    username: str = None,
    match_id: int = None,
) -> subprocess.Popen:
    model_path = model_paths[model_type]
    cmd = [
        'python',
        'human_coach_game.py',
        '--port',
        str(port),
        '--unit-type',
        str(random.randint(0, 5)),
        '--seed',
        str(seed),
        '--model_path',
        model_path,
    ]
    if model_type == 'switch':
        cmd.extend(['--bc-path', model_paths['bc']])
    if save_dir is not None:
        cmd.extend(['--save-dir', save_dir])
    if match_id is not None:
        assert username is not None
        cmd.extend(
            [
                '--username',
                username,
                '--model-type',
                model_type,
                '--match-id',
                str(match_id),
            ]
        )
    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    running_games[port] = p
    while True:
        buf = p.stdout.readline()
        print(buf)
        if buf.decode('utf-8') == 'Waiting for websocket client ...\n':
            break
    def f():
        while True:
            buf = p.stdout.readline().decode('utf-8')
            if buf == '':
                return
            print(buf)
    executor.submit(f)
    executor.submit(check_heartbeat, port, 10)
    return p


@app.on_event('shutdown')
def shutdown():
    print('cleanning running games')
    for p in running_games.values():
        p.terminate()


class Instruction(BaseModel):
    port: int
    inst: str


@app.get('/new_game/{model_type}')
def new_game(model_type: str):
    port = get_available_port()
    global cnt
    start_game(port, cnt, model_type)
    cnt += 1
    return RedirectResponse(f'/minirts_play.html?player_type=spectator&port={port}')


@app.get('/follow')
def game_for_follow(username: str, model_id: int):
    port = get_available_port()
    model_type = users_config[username].id_to_model_type[model_id]
    start_game(
        port,
        cnt,
        model_type,
        save_dir=os.path.join(
            'human_eval', username, 'follow', model_type, uuid.uuid4().hex
        ),
    )
    return RedirectResponse(f'/minirts_play.html?player_type=spectator&port={port}')


class VoteResult(BaseModel):
    username: str
    model_id_list: List[int]


@app.post('/set_vote_result')
def process_vote_result(result: VoteResult):
    model_list = [
        users_config[result.username].id_to_model_type[idx]
        for idx in result.model_id_list
    ]
    print(model_list)
    with env.begin(write=True) as txn:
        key = f'{result.username}/vote'
        txn.put(key.encode(), pickle.dumps(model_list))
    return 'ok'


@app.get('/match')
def game_for_match(username: str, model_id: int, match_id: int):
    # TODO: a match can be played for only once
    model_type = users_config[username].id_to_model_type[model_id]
    with env.begin() as txn:
        tmp = txn.get(f'{username}/match/{model_type}/{match_id}'.encode())
        if tmp != None:
            raise HTTPException(status_code=403, detail='你已经完成了这局游戏，不要 SL 啦')
    port = get_available_port()
    start_game(
        port,
        users_config[username].match_seeds[model_id][match_id],
        model_type,
        save_dir=os.path.join(
            'human_eval', username, 'match', model_type, str(match_id)
        ),
        username=username,
        match_id=match_id,
    )
    return RedirectResponse(f'/minirts_play.html?player_type=spectator&port={port}')


@app.get('/update_match_result')
def update_match_result(username: str, model_type: str, match_id: int, result: str):
    with env.begin(write=True) as txn:
        txn.put(f'{username}/match/{model_type}/{match_id}'.encode(), result.encode())
    return 'ok'


@app.get('/get_finished')
def query_finished(username: str) -> Tuple[List[int], List[List[bool]]]:
    with env.begin() as txn:
        result = []
        for model_id, match_seeds in enumerate(users_config[username].match_seeds):
            tmp = []
            for idx, seed in enumerate(match_seeds):
                model_type = users_config[username].id_to_model_type[model_id]
                tmp.append(
                    txn.get(f'{username}/match/{model_type}/{idx}'.encode()) != None
                )
            result.append(tmp)
        vote_result = txn.get(f'{username}/vote'.encode())
        if vote_result is not None:
            vote_result = pickle.loads(vote_result)
            try:
                tmp = users_config[username].id_to_model_type
                d = dict((x, i) for i, x in enumerate(tmp))
                vote_result_ids = [d[model_type] for model_type in vote_result]
            except KeyError:
                vote_result_ids = None
        else:
            vote_result_ids = None
        return vote_result_ids, result


@app.get('/stop_game/{port}')
def stop_game(port: int):
    if port in running_games:
        p = running_games[port]
        p.terminate()
        running_games.pop(port)
    return 'ok'


@app.get('/heartbeat/{port}')
def heartbeat(port: int):
    timestamps[port] = time.time()
    return 'ok'


@app.post('/pass_inst')
def pass_inst(inst: Instruction):
    timestamps[inst.port] = time.time()
    p = running_games[inst.port]
    p.stdin.write(f'{inst.inst}\n'.encode())
    p.stdin.flush()
    return 'ok'


app.mount(
    '/',
    StaticFiles(directory=os.path.join(os.environ['MINIRTS_ROOT'], 'game/frontend')),
    name='static',
)
