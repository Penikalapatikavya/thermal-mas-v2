"""MAPPO (centralised critic, decentralised actors) for the ground robot + drone in the calibrated 2-D world.
Usage: python3 training/train_mappo.py [--pole 1] [--steps 2000000]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
import torch.nn as nn
from rl_env import RescueEnv, rule_ugv, rule_uav, N_PEOPLE, OBS_DIM, STATE_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--pole", type=int, default=1)
ap.add_argument("--steps", type=int, default=2_000_000)
ap.add_argument("--envs", type=int, default=16)
ap.add_argument("--rollout", type=int, default=256)
ap.add_argument("--randomize", type=int, default=0)
ap.add_argument("--hysteresis", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(0)
torch.set_num_threads(4)
TAG = "_dr" if args.randomize else ""
OUT = os.path.expanduser(f"~/thermal-mas-v2/weights/mappo_pole{args.pole}{TAG}.pt")
LOG = os.path.expanduser(f"~/thermal-mas-v2/outputs/results/mappo_pole{args.pole}{TAG}_training.csv")
N_UGV, N_UAV = N_PEOPLE + 1, 16
GAMMA, LAM, CLIP, EPOCHS, MB, LR, ENT, VF = 0.99, 0.95, 0.2, 5, 1024, 3e-4, 0.01, 0.5


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Actor(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = mlp(OBS_DIM, n, 128)

    def dist(self, obs, mask=None):
        logits = self.net(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)


ugv_actor, uav_actor, critic = Actor(N_UGV), Actor(N_UAV), mlp(STATE_DIM, 1, 256)
params = list(ugv_actor.parameters()) + list(uav_actor.parameters()) + list(critic.parameters())
opt = torch.optim.Adam(params, lr=LR, eps=1e-5)


def evaluate(policy, n=100):
    env = RescueEnv(pole=bool(args.pole), randomize=bool(args.randomize),
                    hysteresis=bool(args.hysteresis) and policy == "rl")
    succ, times, reached = 0, [], 0
    for ep in range(n):
        obs, _ = env.reset(seed=9000 + ep)
        done = False
        while not done:
            if policy == "rule":
                a = (rule_ugv(env), rule_uav(env))
            else:
                with torch.no_grad():
                    m = torch.as_tensor(env.ugv_mask())[None]
                    a0 = ugv_actor.dist(torch.as_tensor(obs[0])[None], m).probs.argmax().item()
                    a1 = uav_actor.dist(torch.as_tensor(obs[1])[None]).probs.argmax().item()
                a = (a0, a1)
            obs, _, _, done = env.step(a)
        reached += len(env.rescued)
        if len(env.rescued) == N_PEOPLE:
            succ += 1
            times.append(max(env.rescue_t.values()))
    return succ / n, (float(np.mean(times)) if times else 999.0), reached / (n * N_PEOPLE)


rs, rt, rr = evaluate("rule")
print(f"rule-based on the 100 evaluation missions: success {rs*100:.0f}%, time {rt:.1f} s, reached {rr*100:.0f}%")
envs = [RescueEnv(pole=bool(args.pole), randomize=bool(args.randomize), hysteresis=bool(args.hysteresis))
        for _ in range(args.envs)]
cur = [e.reset(seed=k) for k, e in enumerate(envs)]
obs = [c[0] for c in cur]
state = [c[1] for c in cur]
ep_seed, best, log_rows, t0 = args.envs, (-1.0, 1e9), [], time.time()
n_updates = args.steps // (args.envs * args.rollout)
for upd in range(1, n_updates + 1):
    B = {k: [] for k in ("o0", "o1", "s", "m", "a0", "a1", "lp", "v", "r", "d")}
    for _ in range(args.rollout):
        o0 = torch.as_tensor(np.stack([o[0] for o in obs]))
        o1 = torch.as_tensor(np.stack([o[1] for o in obs]))
        s = torch.as_tensor(np.stack(state))
        m = torch.as_tensor(np.stack([e.ugv_mask() for e in envs]))
        with torch.no_grad():
            d0, d1 = ugv_actor.dist(o0, m), uav_actor.dist(o1)
            a0, a1 = d0.sample(), d1.sample()
            lp = d0.log_prob(a0) + d1.log_prob(a1)
            v = critic(s).squeeze(-1)
        rew, dn = [], []
        for k, e in enumerate(envs):
            ob, st, r, done = e.step((a0[k].item(), a1[k].item()))
            if done:
                ob, st = e.reset(seed=ep_seed)
                ep_seed += 1
            obs[k], state[k] = ob, st
            rew.append(r)
            dn.append(float(done))
        for key, val in (("o0", o0), ("o1", o1), ("s", s), ("m", m), ("a0", a0), ("a1", a1), ("lp", lp), ("v", v)):
            B[key].append(val)
        B["r"].append(torch.tensor(rew, dtype=torch.float32))
        B["d"].append(torch.tensor(dn, dtype=torch.float32))
    with torch.no_grad():
        last_v = critic(torch.as_tensor(np.stack(state))).squeeze(-1)
    V, R, D = torch.stack(B["v"]), torch.stack(B["r"]), torch.stack(B["d"])
    adv, gae = torch.zeros_like(R), torch.zeros(args.envs)
    for t in reversed(range(args.rollout)):
        nv = last_v if t == args.rollout - 1 else V[t + 1]
        delta = R[t] + GAMMA * nv * (1 - D[t]) - V[t]
        gae = delta + GAMMA * LAM * (1 - D[t]) * gae
        adv[t] = gae
    ret = adv + V
    flat = lambda x: x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1)
    O0, O1, S, M = flat(torch.stack(B["o0"])), flat(torch.stack(B["o1"])), flat(torch.stack(B["s"])), flat(torch.stack(B["m"]))
    A0, A1, LP, ADV, RET = flat(torch.stack(B["a0"])), flat(torch.stack(B["a1"])), flat(torch.stack(B["lp"])), flat(adv), flat(ret)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    N = O0.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(N)
        for i in range(0, N, MB):
            idx = perm[i:i + MB]
            d0, d1 = ugv_actor.dist(O0[idx], M[idx]), uav_actor.dist(O1[idx])
            ratio = torch.exp(d0.log_prob(A0[idx]) + d1.log_prob(A1[idx]) - LP[idx])
            pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[idx]).mean()
            vloss = ((critic(S[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
            ent = (d0.entropy() + d1.entropy()).mean()
            loss = pg + VF * vloss - ENT * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
    if upd % 10 == 0 or upd == n_updates:
        es, et, er = evaluate("rl")
        steps = upd * args.envs * args.rollout
        print(f"update {upd:4d}/{n_updates} ({steps/1e6:.2f}M steps, {time.time()-t0:.0f}s): RL success {es*100:.0f}%, "
              f"time {et:.1f} s, reached {er*100:.0f}%   [rule: {rs*100:.0f}%, {rt:.1f} s]")
        log_rows.append({"update": upd, "steps": steps, "success": es, "time": et, "reached": er})
        if (es, -et) > (best[0], -best[1]):
            best = (es, et)
            torch.save({"ugv": ugv_actor.state_dict(), "uav": uav_actor.state_dict(), "pole": args.pole,
                        "eval": {"success": es, "time": et, "reached": er}}, OUT)
            print(f"   saved new best -> {OUT}")
        import csv
        with open(LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            w.writeheader()
            w.writerows(log_rows)
print(f"best RL: success {best[0]*100:.0f}%, time {best[1]:.1f} s  |  rule: success {rs*100:.0f}%, time {rt:.1f} s")
