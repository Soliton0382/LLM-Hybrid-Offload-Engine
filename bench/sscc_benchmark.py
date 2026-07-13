#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sscc_benchmark.py — SOLITON-SALIENCE CONTEXT COMPRESSION (SSCC)
===============================================================
Benchmark for the Hybrid-Offload-Engine. It answers one honest question a
technical reviewer would actually ask:

    "If I compress the context before feeding the model, does the critical
     information survive — and does the model still answer correctly?
     How many tokens (and KV-cache) do I save?"

Methods compared, at an equal keep-budget:
    full        no compression (oracle / upper bound)
    sscc        Soliton-Salience:  score = cos*(1+a*Amp)*(g+(1-g)*Vel)   [Aurora]
    cosine      pure semantic retrieval  (isolates the contribution of Amp/Vel)
    truncate    naive first-K sliding window (the classic baseline)

Two phases:
    PHASE A  (always, instant, NO model): Monte-Carlo retention on numpy.
    PHASE B  (opt., RUN_MODEL_EVAL=1)   : loads the real GGUF once, compresses,
                                          queries, measures exact-match answer
                                          accuracy + tokens + TTFT + latency +
                                          decode throughput + VRAM.

Two scenarios:
    S1  single-needle among HOMOGENEOUS distractors (same domain) -> salience.
    S2  multi-needle + RECENCY -> isolates Vel (where pure cosine fails).

Dependencies: numpy only (pure-numpy TF-IDF embedder) + llama_cpp (Phase B only).

USAGE:
    source .venv/bin/activate
    python bench/sscc_benchmark.py                            # Phase A (seconds)
    RUN_MODEL_EVAL=1 MODEL_TRIALS=5 python -u bench/sscc_benchmark.py   # + Phase B

ENV: TRIALS(500) N_DISTRACTORS(30) SEED(7) ALPHA(0.35) GAMMA(0.70)
     RATIOS("0.50,0.35,0.25,0.15") RUN_MODEL_EVAL(0) MODEL_TRIALS(5) MAX_TOK(24)
     NO_COLOR(0)  (MODEL_NAME/GPU_LAYERS/CONTEXT_WINDOW inherited from .env)
"""
import os, sys, time, random, re, json
from collections import Counter
from datetime import datetime
import numpy as np

# ------------------------------- ANSI ----------------------------------------
_NC = os.getenv("NO_COLOR", "0") in ("1", "true", "yes") or not sys.stdout.isatty()
def _c(x): return "" if _NC else x
R=_c("\033[0m"); B=_c("\033[1m"); DIM=_c("\033[2m")
RED=_c("\033[91m"); GRN=_c("\033[92m"); YEL=_c("\033[93m")
BLU=_c("\033[94m"); MAG=_c("\033[95m"); CYN=_c("\033[96m"); WHT=_c("\033[97m")

def bar(pct, width=22):
    n=int(round(pct/100*width))
    col=GRN if pct>=90 else (YEL if pct>=50 else RED)
    return col+"█"*n+DIM+"·"*(width-n)+R
def line(w=74): print(DIM+"─"*w+R)
def title(t):
    print(); print(B+CYN+"╔"+"═"*(len(t)+2)+"╗"+R)
    print(B+CYN+"║ "+WHT+t+CYN+" ║"+R); print(B+CYN+"╚"+"═"*(len(t)+2)+"╝"+R)

# ------------------------------- config --------------------------------------
BASE_DIR=os.path.dirname(os.path.abspath(__file__)); REPO_DIR=os.path.dirname(BASE_DIR)
try:
    from dotenv import load_dotenv; load_dotenv(os.path.join(REPO_DIR,".env"))
except Exception: pass
def _f(n,d): return float(os.getenv(n,d))
def _i(n,d): return int(os.getenv(n,d))
def _b(n,d): return os.getenv(n,str(d)) in ("1","true","True","yes")

SEED=_i("SEED",7); TRIALS=_i("TRIALS",500); N_DISTRACT=_i("N_DISTRACTORS",30)
ALPHA=_f("ALPHA",0.35); GAMMA=_f("GAMMA",0.70)
RATIOS=[float(x) for x in os.getenv("RATIOS","0.50,0.35,0.25,0.15").split(",")]
RUN_MODEL=_b("RUN_MODEL_EVAL",0); MODEL_TRIALS=_i("MODEL_TRIALS",5); MAX_TOK=_i("MAX_TOK",24)
random.seed(SEED); np.random.seed(SEED)

# ------------------------------- embedder (pure-numpy TF-IDF) ----------------
_WORD=re.compile(r"[a-z0-9\-]+")
def _tok(s): return _WORD.findall(s.lower())
def embed(docs):
    toks=[_tok(d) for d in docs]; vocab={}
    for t in toks:
        for w in t:
            if w not in vocab: vocab[w]=len(vocab)
    N,V=len(docs),len(vocab); tf=np.zeros((N,V))
    for i,t in enumerate(toks):
        for w,c in Counter(t).items(): tf[i,vocab[w]]=c
    idf=np.log((1.0+N)/(1.0+(tf>0).sum(0)))+1.0
    X=tf*idf; nrm=np.linalg.norm(X,axis=1,keepdims=True); nrm[nrm==0]=1.0
    return X/nrm

def _salience(q,ch):
    E=embed([q]+ch); qv,C=E[0],E[1:]; cos=C@qv
    Amp=np.clip((cos-cos.mean())/(cos.std()+1e-9),0,None)   # relative energy
    Vel=np.linspace(0.2,1.0,len(ch))                        # recency
    return cos*(1+ALPHA*Amp)*(GAMMA+(1-GAMMA)*Vel), cos

def prune(method,q,ch,n_keep):
    n=len(ch); n_keep=max(1,min(n,n_keep))
    if method=="full":     return list(range(n))
    if method=="truncate": return list(range(n_keep))       # classic first-K
    s,cos=_salience(q,ch)
    return sorted(np.argsort(-(s if method=="sscc" else cos))[:n_keep].tolist())

# ------------------------------- dataset -------------------------------------
SECRET="ZX-4471-DELTA"
QUERY="What is the reactor core activation code?"
NEEDLES=[
    "The reactor core activation code is {s}, confirmed by the chief engineer.",
    "Per the logbook, the reactor core activation code is {s} as of today.",
    "Security note: reactor core activation code {s} was issued this morning.",
    "The current reactor core activation code has been set to {s}.",
    "Control room confirmed the reactor core activation code is {s}.",
]
DISTRACT=[
    "The reactor coolant system was inspected during routine maintenance.",
    "Reactor core temperature remained within nominal limits all week.",
    "The engineer reviewed reactor safety protocols before the night shift.",
    "Activation of the backup reactor pump was tested successfully.",
    "The reactor control room displayed stable pressure across all sensors.",
    "Core shielding was reinforced following the annual safety audit.",
    "The reactor startup sequence requires authorization from two staff.",
    "Cooling towers near the reactor were cleaned to improve exchange.",
    "The reactor logbook recorded no anomalies during the quarter.",
    "Engineers calibrated the reactor neutron flux detectors on Tuesday.",
    "The reactor turbine hall was repainted during the shutdown.",
    "Reactor fuel rods were replaced per the maintenance schedule.",
    "The activation panel for reactor lighting was upgraded recently.",
    "Reactor vibration levels were monitored by the diagnostic subsystem.",
    "The reactor emergency exits were re-signed to meet safety code.",
    "Staff completed reactor evacuation drills without reported issues.",
    "The reactor water purity index stayed above threshold.",
    "A new reactor telemetry dashboard was deployed to operators.",
    "Reactor perimeter access badges were reissued to personnel.",
    "The reactor documentation archive was migrated to a new server.",
    "The reactor ventilation filters were swapped during maintenance.",
    "Reactor pressure vessel welds passed ultrasonic inspection.",
    "The reactor training simulator received a firmware update.",
    "Auxiliary reactor generators were load-tested by the team.",
    "Reactor coolant flow rates were logged hourly by technicians.",
    "The reactor spare-parts inventory was reconciled centrally.",
    "Reactor radiation monitors were recalibrated to the reference.",
    "Reactor containment doors were serviced by the mechanical crew.",
    "The reactor site perimeter fence was inspected for damage.",
    "Reactor auxiliary lighting circuits were tested by electricians.",
]
def make_ctx(nd):
    pool=DISTRACT[:]; random.shuffle(pool); ch=pool[:max(1,nd)]
    pos=random.randint(0,len(ch)); ch.insert(pos,random.choice(NEEDLES).format(s=SECRET))
    return ch,pos

# =============================================================================
#  PHASE A
# =============================================================================
def phase_a():
    title("PHASE A · RETENTION vs COMPRESSION  (numpy · no model)")
    print(f"{DIM}Task: hide the answer among {N_DISTRACT} HOMOGENEOUS distractors, "
          f"compress, check if the needle survives · {TRIALS} trials{R}\n")
    methods=[("full","full",WHT),("SSCC","sscc",GRN),
             ("cosine","cosine",BLU),("truncate","truncate",MAG)]
    data={}
    for r in RATIOS:
        print(f"{B}{WHT}▸ Context kept: {int(r*100)}%{R}  {DIM}(dropping {int((1-r)*100)}%){R}")
        row={}
        for lbl,m,col in methods:
            hit=0; st=random.getstate()
            for _ in range(TRIALS):
                ch,pos=make_ctx(N_DISTRACT); nk=max(1,round(len(ch)*r))
                if pos in set(prune(m,QUERY,ch,nk)): hit+=1
            random.setstate(st); pct=100*hit/TRIALS; row[m]=pct
            tag="" if m=="full" else (f"  {GRN}✔ keeps{R}" if pct>=99 else
                 (f"  {YEL}~ drops{R}" if pct>=50 else f"  {RED}✗ FAILS{R}"))
            print(f"   {col}{lbl:<9}{R} {bar(pct)} {col}{pct:5.1f}%{R}{tag}")
        data[r]=row; print()
    worst=min(RATIOS)
    line()
    print(f"{B}SUMMARY{R}  at MAX compression ({int(worst*100)}% context):")
    print(f"   {GRN}SSCC{R}: {GRN}{data[worst]['sscc']:.0f}%{R}   "
          f"{BLU}cosine{R}: {data[worst]['cosine']:.0f}%   "
          f"{MAG}truncate (classic){R}: {RED}{data[worst]['truncate']:.0f}%{R}")
    return data

def phase_a_vel():
    title("PHASE A-bis · Vel (recency) CONTRIBUTION  ·  CURRENT vs STALE fact")
    print(f"{DIM}Two codes in memory (old + current). Minimal budget -> the system MUST "
          f"choose. Only recency (Vel) picks the correct one.{R}\n")
    def ctx2(nd):
        old="AA-1111-ALPHA"; new=SECRET
        no=f"The reactor core activation code is {old}."
        nn=f"The reactor core activation code is {new}."
        pool=DISTRACT[:]; random.shuffle(pool); body=pool[:max(2,nd)]
        body.insert(random.randint(0,len(body)//2),no)
        body.insert(random.randint(int(len(body)*0.8),len(body)),nn)
        return body, body.index(nn)
    q="What is the CURRENT reactor core activation code?"
    data={}
    for nk in (1,2,3):
        print(f"{B}{WHT}▸ Budget = {nk} fact(s) in memory{R}"); row={}
        for lbl,m,col in [("SSCC","sscc",GRN),("cosine","cosine",BLU)]:
            hit=0; st=random.getstate()
            for _ in range(TRIALS):
                ch,pos=ctx2(N_DISTRACT)
                if pos in set(prune(m,q,ch,nk)): hit+=1
            random.setstate(st); pct=100*hit/TRIALS; row[m]=pct
            print(f"   {col}{lbl:<8}{R} {bar(pct)} {col}{pct:5.1f}%{R}")
        data[nk]=row; print()
    d=data[1]['sscc']-data[1]['cosine']
    print(f"{B}➜ At budget=1 only recency (Vel) finds the CURRENT code: "
          f"{GRN}+{d:.0f}pp over standard retrieval{R}.")
    return data

# =============================================================================
#  PHASE B — real GGUF
# =============================================================================
def phase_b():
    title("PHASE B · REAL TASK on the GGUF  ·  accuracy + speed + tokens")
    MODEL_NAME=os.getenv("MODEL_NAME","model.gguf")
    MODEL_PATH=os.path.join(REPO_DIR,"models",MODEL_NAME)
    GPU_LAYERS=_i("GPU_LAYERS",25); CTX=_i("CONTEXT_WINDOW",4096)
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}[X] Model not found: {MODEL_PATH}{R}\n    Skipping Phase B."); return None
    import subprocess, psutil
    from llama_cpp import Llama
    def vram():
        try:
            o=subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5).stdout.split()
            return int(o[0]) if o else None
        except Exception: return None
    PHYS=psutil.cpu_count(logical=False) or 8
    print(f"{DIM}model {MODEL_NAME} · gpu_layers {GPU_LAYERS} · ctx {CTX} · {PHYS} cores{R}")
    v0=vram(); t0=time.time()
    llm=Llama(model_path=MODEL_PATH,n_gpu_layers=GPU_LAYERS,n_threads=PHYS,
              n_ctx=CTX,flash_attn=True,use_mmap=True,verbose=False)
    v1=vram()
    print(f"{GRN}✔ load {time.time()-t0:.1f}s{R} · VRAM {v0}→{v1} MiB "
          f"(hybrid offload, {(v1-v0) if (v0 and v1) else '?'} MiB on GPU)\n")
    SYS=("You are precise. Answer ONLY from the context. "
         "Reply with just the activation code, nothing else.")
    def ask(chunks):
        ctx="\n".join(f"- {c}" for c in chunks)
        prompt=(f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nContext:\n{ctx}\n\n"
                f"Question: {QUERY}<|im_end|>\n<|im_start|>assistant\n")
        toks=llm.tokenize(prompt.encode()); n_in=len(toks)
        t_start=time.time(); first=None; n_out=0; buf=[]
        for ch in llm.create_completion(prompt=toks,max_tokens=MAX_TOK,temperature=0.0,
                                        stream=True,stop=["<|im_end|>","<|endoftext|>"]):
            txt=ch["choices"][0]["text"]
            if txt:
                if first is None: first=time.time()
                n_out+=1; buf.append(txt)
        tend=time.time()
        ttft=(first-t_start) if first else 0.0
        dtps=n_out/(tend-first) if first and (tend-first)>0 else 0.0
        return "".join(buf),n_in,n_out,ttft,(tend-t_start),dtps
    def ok(a): return SECRET in a.upper().replace(" ","").replace("_","-")
    methods=[("full","full",WHT),("SSCC","sscc",GRN),("truncate","truncate",MAG)]
    print(f"{DIM}Task: hide '{SECRET}' among {N_DISTRACT} distractors, compress, ask. "
          f"{MODEL_TRIALS} trials/method/ratio.{R}\n")
    hdr=(f"{B}{'keep':>6} {'method':>9} {'acc':>6} {'tok_in':>7} "
         f"{'saved':>7} {'TTFT ms':>8} {'lat s':>7} {'dec t/s':>8}{R}")
    results={}
    for r in RATIOS:
        print(f"{B}{WHT}▸ context {int(r*100)}%{R}"); print(hdr); line()
        full_tin=None
        for lbl,m,col in methods:
            acc=0; tins=[]; ttfts=[]; lats=[]; dtps=[]; st=random.getstate()
            for _ in range(MODEL_TRIALS):
                ch,pos=make_ctx(N_DISTRACT); nk=max(1,round(len(ch)*r))
                keep=prune(m,QUERY,ch,nk); sub=[ch[i] for i in keep]
                a,n_in,n_out,ttft,lat,dt=ask(sub)
                acc+=ok(a); tins.append(n_in); ttfts.append(ttft*1000); lats.append(lat); dtps.append(dt)
            random.setstate(st); mean_tin=np.mean(tins)
            if m=="full": full_tin=mean_tin
            saved=100*(1-mean_tin/full_tin) if full_tin else 0.0
            a_pct=100*acc/MODEL_TRIALS
            acol=GRN if a_pct>=99 else (YEL if a_pct>=50 else RED)
            results.setdefault(f"{r}",{})[m]=dict(acc=a_pct,tok_in=float(mean_tin),
                saved=saved,ttft_ms=float(np.median(ttfts)),lat=float(np.median(lats)),
                dec_tps=float(np.median(dtps)))
            print(f"{col}{int(r*100):>5}% {lbl:>9}{R} {acol}{a_pct:>5.0f}%{R} "
                  f"{mean_tin:>7.0f} {saved:>6.1f}% {np.median(ttfts):>7.0f} "
                  f"{np.median(lats):>6.2f} {np.median(dtps):>7.2f}")
        print()
    worst=f"{min(RATIOS)}"; rw=results.get(worst,{})
    line()
    print(f"{B}🏆 LEADERBOARD at MAX compression ({int(min(RATIOS)*100)}% context){R}")
    order=sorted(rw.items(),key=lambda kv:(-kv[1]['acc'],kv[1]['tok_in']))
    medals=["🥇","🥈","🥉","  "]
    for i,(m,d) in enumerate(order):
        col=GRN if m=="sscc" else (WHT if m=="full" else MAG)
        print(f"   {medals[min(i,3)]} {col}{m:<9}{R} acc {d['acc']:>3.0f}% · "
              f"tok {d['tok_in']:>4.0f} · saved {d['saved']:>4.1f}% · {d['dec_tps']:.1f} t/s")
    return results

# =============================================================================
def main():
    print(B+MAG+"\n"+"█"*74+R)
    print(B+WHT+"  SSCC · SOLITON-SALIENCE CONTEXT COMPRESSION — BENCHMARK"+R)
    print(DIM+f"  seed={SEED} · trials={TRIALS} · distractors={N_DISTRACT} · "
          f"α={ALPHA} γ={GAMMA} · model_eval={'ON' if RUN_MODEL else 'OFF'}"+R)
    print(B+MAG+"█"*74+R)
    out={"meta":{"ts":datetime.now().isoformat(timespec="seconds"),"seed":SEED,
                 "trials":TRIALS,"n_distract":N_DISTRACT,"alpha":ALPHA,"gamma":GAMMA}}
    out["phase_a"]=phase_a()
    out["phase_a_vel"]=phase_a_vel()
    if RUN_MODEL: out["phase_b"]=phase_b()
    rd=os.path.join(BASE_DIR,"results"); os.makedirs(rd,exist_ok=True)
    fp=os.path.join(rd,f"sscc_{datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump(out,open(fp,"w"),indent=2,default=float)
    print(f"\n{DIM}[+] JSON: {fp}{R}")
    print(f"{B}{GRN}[+] Done. The shell has spoken.{R}\n")

if __name__=="__main__":
    main()
