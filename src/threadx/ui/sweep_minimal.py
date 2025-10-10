# src/threadx/ui/sweep.py
import tkinter as tk
from tkinter import ttk, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time


def create_sweep_page(parent, indicator_bank) -> ttk.Frame:
    frame = ttk.Frame(parent)
    cfg = ttk.LabelFrame(frame, text="Grille paramétrique (démo)")
    cfg.pack(fill=tk.X, padx=10, pady=10)

    tk.Label(cfg, text="BB std: 1.5 → 3.0 pas 0.1 | MA window: 10 → 60 pas 5").pack(
        anchor="w", padx=6
    )
    run_btn = ttk.Button(cfg, text="Run Sweep")
    run_btn.pack(side=tk.LEFT, padx=6)
    prog = ttk.Progressbar(cfg, mode="indeterminate")
    prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    table = ttk.Treeview(
        frame,
        columns=("std", "window", "pnl", "sharpe", "maxdd", "pf"),
        show="headings",
    )
    for c in ("std", "window", "pnl", "sharpe", "maxdd", "pf"):
        table.heading(c, text=c)
        table.column(c, width=100, anchor="center")
    table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    pool = ThreadPoolExecutor(max_workers=4)

    def run_job():
        run_btn.config(state=tk.DISABLED)
        prog.start(12)
        table.delete(*table.get_children())

        stds = np.round(np.arange(1.5, 3.01, 0.1), 2)
        wins = list(range(10, 61, 5))
        combos = [(s, w) for s in stds for w in wins]

        futures = []
        for s, w in combos:
            futures.append(pool.submit(_fake_eval, s, w))
        done = 0

        def poll():
            nonlocal done
            while futures and futures[0].done():
                r = futures.pop(0).result()
                done += 1
                table.insert("", "end", values=r)
            if futures:
                frame.after(80, poll)
            else:
                prog.stop()
                run_btn.config(state=tk.NORMAL)

        poll()

    def _fake_eval(std, win):
        # Ici tu brancheras bank.ensure → engine.run → performance.summarize
        time.sleep(0.02)
        pnl = np.random.uniform(-0.1, 0.25)
        sharpe = np.random.uniform(0.5, 2.0)
        maxdd = -np.random.uniform(0.01, 0.12)
        pf = np.random.uniform(0.8, 2.5)
        return (
            std,
            win,
            round(pnl, 4),
            round(sharpe, 3),
            round(maxdd, 4),
            round(pf, 3),
        )

    run_btn.config(command=run_job)
    return frame
