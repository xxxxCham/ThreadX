"""
ThreadX Streamlit Fallback App - Phase 8
========================================

Minimal Streamlit application as fallback UI for ThreadX.

This is a secondary interface to the main Tkinter application,
providing basic functionality for parameter configuration,
backtest execution, and results visualization.

Note: Tkinter app.py is the primary interface. This Streamlit app
serves as a fallback for quick prototyping and web-based access.

Features:
- Parameter configuration and validation
- Basic backtest execution via Engine Phase 5
- Results visualization with charts and tables
- Export functionality
- Real-time logging display

Author: ThreadX Framework
Version: Phase 8 - UI Components (Fallback)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Configure page
st.set_page_config(
    page_title="ThreadX - Algorithmic Trading Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ThreadX imports - with authentication support
try:
    from threadx.config.settings import Settings
    from threadx.config.auth import get_auth_manager, AuthManager
    from threadx.data.client import get_data_client, DataClient
    from threadx.utils.log import get_logger, setup_logging_once
    from threadx.data.bank import Bank
    from threadx.engine.backtest import BacktestEngine
    from threadx.performance.metrics import PerformanceCalculator
    from threadx.ui.charts import plot_equity, plot_drawdown
    from threadx.ui.tables import render_trades_table, render_metrics_table, export_table
    
    # Initialiser les gestionnaires d'authentification et de données
    AUTH_MANAGER = get_auth_manager()
    DATA_CLIENT = get_data_client()
    THREADX_AVAILABLE = True
except ImportError:
    # Mock implementations for fallback
    st.warning("⚠️ ThreadX modules not fully available. Running in mock mode.")
    
    AUTH_MANAGER = None
    DATA_CLIENT = None 
    THREADX_AVAILABLE = False
    
    class Settings:
        @staticmethod
        def get(key: str, default=None):
            return default
    
    def get_logger(name: str):
        return logging.getLogger(name)
    
    def setup_logging_once():
        logging.basicConfig(level=logging.INFO)
    
    class Bank:
        def ensure(self, *args, **kwargs):
            time.sleep(1)
            return True
    
    class BacktestEngine:
        def run(self, *args, **kwargs):
            time.sleep(2)
            dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
            returns = pd.Series(np.random.randn(1000) * 0.01, index=dates)
            trades = pd.DataFrame({
                'entry_time': dates[::50],
                'exit_time': dates[50::50],
                'pnl': np.random.randn(20) * 100,
                'side': ['LONG'] * 20,
                'entry_price': 50000 + np.random.randn(20) * 1000,
                'exit_price': 50000 + np.random.randn(20) * 1000
            })
            return returns, trades
    
    class PerformanceCalculator:
        @staticmethod
        def summarize(returns, trades):
            return {
                'final_equity': 11000,
                'total_return': 0.10,
                'cagr': 0.12,
                'sharpe': 1.5,
                'max_drawdown': -0.05,
                'total_trades': len(trades),
                'win_rate': 0.6,
                'profit_factor': 1.8,
                'sortino': 1.2,
                'annual_volatility': 0.15
            }
    
    def plot_equity(equity, save_path=None):
        return save_path
    
    def plot_drawdown(equity, save_path=None):
        return save_path
    
    def render_trades_table(trades):
        return {'data': trades, 'summary': {}}
    
    def render_metrics_table(metrics):
        return {'data': pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])}
    
    def export_table(df, path):
        return path


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'bank': Bank(),
        'engine': BacktestEngine(),
        'performance': PerformanceCalculator(),
        'last_results': None,
        'last_returns': None,
        'last_trades': None,
        'last_metrics': None,
        'logs': [],
        'indicators_generated': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def authentication_tab():
    """Authentication and API credentials management tab."""
    st.header("🔐 API Authentication")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Configuration Status")
        
        if THREADX_AVAILABLE and AUTH_MANAGER:
            # Vérifier le statut des credentials
            auth_status = AUTH_MANAGER.validate_all_credentials()
            
            st.markdown("**Variables d'environnement configurées:**")
            
            # Binance
            binance_status = "✅ Configuré" if auth_status.get('binance', False) else "❌ Manquant"
            st.markdown(f"- **Binance API**: {binance_status}")
            if auth_status.get('binance', False):
                st.success("Clés Binance détectées - Limites API étendues disponibles")
            else:
                st.warning("Clés Binance manquantes - Utilisation des endpoints publics uniquement")
            
            # CoinGecko
            coingecko_status = "✅ Configuré" if auth_status.get('coingecko', False) else "❌ Manquant"
            st.markdown(f"- **CoinGecko API**: {coingecko_status}")
            if auth_status.get('coingecko', False):
                st.success("Clé CoinGecko détectée - Limites étendues (500 req/min)")
            else:
                st.info("Clé CoinGecko manquante - Limites publiques (10-50 req/min)")
            
            # Autres APIs
            for provider in ['alpha_vantage', 'polygon', 'finnhub']:
                status = "✅ Configuré" if auth_status.get(provider, False) else "❌ Manquant"
                st.markdown(f"- **{provider.replace('_', ' ').title()}**: {status}")
        
        else:
            st.warning("⚠️ Module d'authentification non disponible en mode mock")
            st.markdown("""
            **Variables d'environnement à configurer:**
            - `BINANCE_API_KEY` : Clé API Binance
            - `BINANCE_API_SECRET` : Secret API Binance  
            - `COINGECKO_API_KEY` : Clé API CoinGecko (optionnelle)
            - `ALPHA_VANTAGE_API_KEY` : Clé API Alpha Vantage (optionnelle)
            - `POLYGON_API_KEY` : Clé API Polygon (optionnelle)
            """)
    
    with col2:
        st.subheader("🔧 Configuration")
        
        st.markdown("**Comment configurer les clés API:**")
        
        # Instructions pour Windows
        with st.expander("🪟 Configuration Windows"):
            st.code("""
# PowerShell (recommandé)
$env:BINANCE_API_KEY="votre_cle_api_binance"
$env:BINANCE_API_SECRET="votre_secret_binance"
$env:COINGECKO_API_KEY="votre_cle_coingecko"

# Ou via Variables d'environnement système
# 1. Win+R → sysdm.cpl
# 2. Onglet "Avancé" → Variables d'environnement
# 3. Ajouter les variables utilisateur
            """, language="powershell")
        
        # Instructions pour Linux/Mac
        with st.expander("🐧 Configuration Linux/Mac"):
            st.code("""
# Bash/Zsh
export BINANCE_API_KEY="votre_cle_api_binance"
export BINANCE_API_SECRET="votre_secret_binance"
export COINGECKO_API_KEY="votre_cle_coingecko"

# Ajouter au ~/.bashrc ou ~/.zshrc pour persistance
echo 'export BINANCE_API_KEY="votre_cle"' >> ~/.bashrc
            """, language="bash")
        
        # Instructions pour Docker
        with st.expander("🐳 Configuration Docker"):
            st.code("""
# docker-compose.yml
environment:
  - BINANCE_API_KEY=votre_cle_api_binance
  - BINANCE_API_SECRET=votre_secret_binance
  - COINGECKO_API_KEY=votre_cle_coingecko

# Ou fichier .env
docker run --env-file .env votre_image
            """, language="yaml")
    
    st.markdown("---")
    
    # Section test des connexions
    st.subheader("🧪 Test des Connexions API")
    
    col_test1, col_test2 = st.columns([1, 1])
    
    with col_test1:
        if st.button("🔍 Tester Toutes les APIs", type="primary"):
            if THREADX_AVAILABLE and DATA_CLIENT:
                with st.spinner("Test des connexions en cours..."):
                    try:
                        test_results = DATA_CLIENT.test_all_connections()
                        
                        st.subheader("📊 Résultats des Tests")
                        for provider, status in test_results.items():
                            if status:
                                st.success(f"✅ {provider.capitalize()}: Connexion OK")
                            else:
                                st.error(f"❌ {provider.capitalize()}: Échec de connexion")
                        
                        # Statistiques globales
                        success_count = sum(test_results.values())
                        total_count = len(test_results)
                        success_rate = success_count / total_count * 100
                        
                        st.metric(
                            "Taux de Succès", 
                            f"{success_rate:.1f}%", 
                            f"{success_count}/{total_count} APIs"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors du test: {str(e)}")
            else:
                st.error("❌ Module de données non disponible en mode mock")
    
    with col_test2:
        if st.button("📊 Afficher Statut Détaillé"):
            if THREADX_AVAILABLE and AUTH_MANAGER:
                # Capture du statut d'authentification
                import io
                from contextlib import redirect_stdout
                
                string_buffer = io.StringIO()
                with redirect_stdout(string_buffer):
                    AUTH_MANAGER.print_auth_status()
                
                status_output = string_buffer.getvalue()
                st.code(status_output, language=None)
            else:
                st.error("❌ Gestionnaire d'authentification non disponible en mode mock")
    
    # Section sécurité
    st.markdown("---")
    st.subheader("🔒 Notes de Sécurité")
    
    st.info("""
    **Bonnes Pratiques de Sécurité:**
    
    ✅ **Recommandé:**
    - Utiliser des variables d'environnement système
    - Ne jamais commiter les clés dans le code source
    - Utiliser des clés API dédiées au trading (avec permissions limitées)
    - Activer l'authentification IP sur les APIs supportées
    
    ⚠️ **Éviter:**
    - Stocker les clés dans des fichiers de configuration
    - Partager les clés via des canaux non sécurisés
    - Utiliser des clés avec permissions de retrait
    """)
    
    # Section links utiles
    with st.expander("🔗 Liens Utiles - Obtenir vos Clés API"):
        st.markdown("""
        **Binance:**
        - [Créer une clé API Binance](https://www.binance.com/en/my/settings/api-management)
        - [Documentation API Binance](https://binance-docs.github.io/apidocs/)
        
        **CoinGecko:**
        - [Obtenir une clé CoinGecko](https://www.coingecko.com/en/api/pricing)
        - [Documentation API CoinGecko](https://www.coingecko.com/en/api/documentation)
        
        **Alpha Vantage:**
        - [Clé gratuite Alpha Vantage](https://www.alphavantage.co/support/#api-key)
        
        **Polygon:**
        - [Inscription Polygon](https://polygon.io/)
        """)


def main():
    """Main Streamlit application."""
    # Setup
    setup_logging_once()
    logger = get_logger(__name__)
    init_session_state()
    
    # Header
    st.title("📈 ThreadX - Algorithmic Trading Framework")
    st.markdown("---")
    
    # Sidebar notice
    with st.sidebar:
        st.info("💡 **Note**: This is a fallback interface. The main ThreadX UI is the Tkinter application.")
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Reset All Data", type="secondary"):
            for key in ['last_results', 'last_returns', 'last_trades', 'last_metrics']:
                st.session_state[key] = None
            st.session_state.indicators_generated = False
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔐 Authentication", "📊 Data & Indicators", "⚙️ Strategy", "🚀 Backtest", "📈 Performance", "📋 Logs"])
    
    with tab1:
        authentication_tab()
    
    with tab2:
        data_indicators_tab()
    
    with tab3:
        strategy_tab()
    
    with tab4:
        backtest_tab()
    
    with tab5:
        performance_tab()
    
    with tab6:
        logs_tab()


def data_indicators_tab():
    """Data and Indicators configuration tab."""
    st.header("📊 Data & Indicators")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Configuration")
        
        symbol = st.selectbox("Symbol", ["BTCUSDC", "ETHUSD", "ADAUSD", "SOLUSD"], index=0)
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=2)
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01").date())
        with col_date2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31").date())
        
        # Data info
        st.info(f"📅 Selected: {symbol} {timeframe} from {start_date} to {end_date}")
    
    with col2:
        st.subheader("Indicator Configuration")
        
        bb_period = st.number_input("Bollinger Bands Period", min_value=5, max_value=100, value=20)
        bb_std = st.number_input("Bollinger Bands Std", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        atr_period = st.number_input("ATR Period", min_value=5, max_value=50, value=14)
        
        st.markdown("---")
        
        if st.button("🔧 Regenerate Indicators", type="primary"):
            with st.spinner("Regenerating indicators..."):
                try:
                    result = st.session_state.bank.ensure(
                        symbol=symbol,
                        timeframe=timeframe,
                        bb_period=bb_period,
                        bb_std=bb_std,
                        atr_period=atr_period
                    )
                    
                    if result:
                        st.session_state.indicators_generated = True
                        st.success("✅ Indicators regenerated successfully!")
                        st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] Indicators regenerated for {symbol} {timeframe}")
                    else:
                        st.error("❌ Indicator regeneration failed")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Indicator status
    if st.session_state.indicators_generated:
        st.success("✅ Indicators are up to date")
    else:
        st.warning("⚠️ Indicators need to be regenerated")


def strategy_tab():
    """Strategy parameters configuration tab."""
    st.header("⚙️ Strategy Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parameter Management")
        
        # File upload for JSON parameters
        uploaded_file = st.file_uploader("Upload Parameter JSON", type=['json'])
        if uploaded_file is not None:
            try:
                params = json.load(uploaded_file)
                st.success("✅ Parameters loaded from JSON")
                st.json(params)
            except Exception as e:
                st.error(f"❌ Error loading JSON: {e}")
        
        # Download template
        template_params = {
            "symbol": "BTCUSDC",
            "timeframe": "15m",
            "bb_period": 20,
            "bb_std": 2.0,
            "entry_z": 2.0,
            "k_sl": 1.5,
            "leverage": 3,
            "risk": 0.02,
            "trail_k": 1.0
        }
        
        st.download_button(
            "📥 Download Parameter Template",
            data=json.dumps(template_params, indent=2),
            file_name="threadx_params_template.json",
            mime="application/json"
        )
    
    with col2:
        st.subheader("Strategy Parameters")
        
        # Entry parameters
        entry_z = st.number_input("Entry Z-Score", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        k_sl = st.number_input("Stop Loss K", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        trail_k = st.number_input("Trail K", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        
        st.markdown("---")
        
        # Risk parameters
        leverage = st.number_input("Leverage", min_value=1, max_value=20, value=3)
        risk = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) / 100
        
        # Validation
        if st.button("✅ Validate Parameters"):
            params = {
                'entry_z': entry_z,
                'k_sl': k_sl,
                'trail_k': trail_k,
                'leverage': leverage,
                'risk': risk
            }
            
            valid, message = validate_parameters(params)
            if valid:
                st.success(f"✅ Parameters are valid: {message}")
            else:
                st.error(f"❌ Invalid parameters: {message}")
    
    # Current parameters display
    st.subheader("Current Parameters")
    current_params = {
        'entry_z': entry_z,
        'k_sl': k_sl,
        'trail_k': trail_k,
        'leverage': leverage,
        'risk': risk
    }
    
    st.json(current_params)


def backtest_tab():
    """Backtest execution tab."""
    st.header("🚀 Backtest Execution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Execution Settings")
        
        # Get parameters from strategy tab (stored in session state or defaults)
        params = {
            'symbol': 'BTCUSDC',
            'timeframe': '15m',
            'entry_z': 2.0,
            'bb_period': 20,
            'bb_std': 2.0,
            'k_sl': 1.5,
            'leverage': 3,
            'risk': 0.02,
            'trail_k': 1.0
        }
        
        # Options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            use_gpu = st.checkbox("🚀 Use GPU Acceleration", value=False)
        with col_opt2:
            cache_indicators = st.checkbox("💾 Cache Indicators", value=True)
        
        # Execute backtest
        if st.button("▶️ Run Backtest", type="primary", use_container_width=True):
            if not st.session_state.indicators_generated:
                st.warning("⚠️ Please regenerate indicators first")
                return
            
            with st.spinner("Running backtest..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("Loading data...")
                        elif i < 60:
                            status_text.text("Calculating indicators...")
                        elif i < 90:
                            status_text.text("Executing strategy...")
                        else:
                            status_text.text("Generating results...")
                        time.sleep(0.02)  # Simulate work
                    
                    # Call engine
                    returns, trades = st.session_state.engine.run(
                        **params,
                        use_gpu=use_gpu,
                        cache_indicators=cache_indicators
                    )
                    
                    # Calculate performance
                    metrics = st.session_state.performance.summarize(returns, trades)
                    
                    # Store results
                    st.session_state.last_returns = returns
                    st.session_state.last_trades = trades
                    st.session_state.last_metrics = metrics
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Show success
                    st.success(f"✅ Backtest completed! {len(trades)} trades executed.")
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] Backtest completed: {len(trades)} trades")
                    
                    # Show quick stats
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                    with col_stat2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.3f}")
                    with col_stat3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    with col_stat4:
                        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")
                    st.session_state.logs.append(f"[{time.strftime('%H:%M:%S')}] Backtest failed: {str(e)}")
    
    with col2:
        st.subheader("Quick Stats")
        
        if st.session_state.last_metrics:
            metrics = st.session_state.last_metrics
            
            st.metric("Final Equity", f"${metrics.get('final_equity', 0):,.2f}")
            st.metric("Total Trades", f"{metrics.get('total_trades', 0):,}")
            st.metric("CAGR", f"{metrics.get('cagr', 0):.2%}")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}")
            
        else:
            st.info("Run a backtest to see statistics")


def performance_tab():
    """Performance analysis and visualization tab."""
    st.header("📈 Performance Analysis")
    
    if not has_results():
        st.info("🔍 Run a backtest to see performance analysis")
        return
    
    returns = st.session_state.last_returns
    trades = st.session_state.last_trades
    metrics = st.session_state.last_metrics
    
    # Metrics overview
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Equity", f"${metrics.get('final_equity', 0):,.2f}")
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
    with col2:
        st.metric("CAGR", f"{metrics.get('cagr', 0):.2%}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        st.metric("Sortino Ratio", f"{metrics.get('sortino', 0):.3f}")
    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2%}")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}")
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Equity Curve")
        
        # Calculate equity curve
        equity = (1 + returns).cumprod() * 10000  # Start with $10,000
        
        # Create chart data
        chart_data = pd.DataFrame({
            'Date': equity.index,
            'Equity': equity.values
        })
        
        st.line_chart(chart_data.set_index('Date'))
        
        # Export charts
        if st.button("💾 Export Charts"):
            try:
                output_dir = Path("exports")
                output_dir.mkdir(exist_ok=True)
                
                equity_path = plot_equity(equity, save_path=output_dir / "equity_curve.png")
                drawdown_path = plot_drawdown(equity, save_path=output_dir / "drawdown.png")
                
                st.success(f"✅ Charts exported to {output_dir}")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    with col2:
        st.subheader("📉 Drawdown")
        
        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max - 1) * 100
        
        drawdown_data = pd.DataFrame({
            'Date': drawdown.index,
            'Drawdown': drawdown.values
        })
        
        st.area_chart(drawdown_data.set_index('Date'), color='#ff6b6b')
    
    # Trades table
    st.subheader("📋 Trades Analysis")
    
    if not trades.empty:
        # Trade summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trades", len(trades))
        with col2:
            winning_trades = len(trades[trades['pnl'] > 0])
            st.metric("Winning Trades", winning_trades)
        with col3:
            st.metric("Average PnL", f"${trades['pnl'].mean():.2f}")
        
        # Trades table with filters
        st.markdown("**Filter Trades:**")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            side_filter = st.selectbox("Side", ["All", "LONG", "SHORT"])
        with col_filter2:
            min_pnl = st.number_input("Min PnL", value=float(trades['pnl'].min()))
        with col_filter3:
            max_pnl = st.number_input("Max PnL", value=float(trades['pnl'].max()))
        
        # Apply filters
        filtered_trades = trades.copy()
        if side_filter != "All":
            filtered_trades = filtered_trades[filtered_trades['side'] == side_filter]
        filtered_trades = filtered_trades[
            (filtered_trades['pnl'] >= min_pnl) & 
            (filtered_trades['pnl'] <= max_pnl)
        ]
        
        st.dataframe(filtered_trades, use_container_width=True)
        
        # Export trades
        if st.button("💾 Export Trades Data"):
            try:
                output_path = Path("exports/trades_data.csv")
                output_path.parent.mkdir(exist_ok=True)
                
                export_table(filtered_trades, output_path)
                st.success(f"✅ Trades exported to {output_path}")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    else:
        st.info("No trades data available")


def logs_tab():
    """Logs display and management tab."""
    st.header("📋 Application Logs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Log level filter
        log_level = st.selectbox("Log Level", ["All", "INFO", "WARNING", "ERROR"], index=0)
    
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.logs = []
            st.rerun()
    
    # Display logs
    if st.session_state.logs:
        # Filter logs by level
        filtered_logs = st.session_state.logs
        if log_level != "All":
            filtered_logs = [log for log in st.session_state.logs if log_level in log]
        
        # Display in code block
        log_text = "\n".join(filtered_logs)
        st.code(log_text, language=None)
        
        # Export logs
        if st.button("💾 Export Logs"):
            try:
                output_path = Path("exports/threadx_logs.txt")
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, 'w') as f:
                    f.write(log_text)
                
                st.success(f"✅ Logs exported to {output_path}")
                
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
    
    else:
        st.info("No logs available")


# Helper functions

def validate_parameters(params: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate strategy parameters."""
    try:
        if params['entry_z'] <= 0 or params['entry_z'] > 5:
            return False, "Entry Z must be between 0 and 5"
        
        if params['k_sl'] <= 0 or params['k_sl'] > 5:
            return False, "Stop Loss K must be between 0 and 5"
        
        if params['leverage'] <= 0 or params['leverage'] > 20:
            return False, "Leverage must be between 1 and 20"
        
        if params['risk'] <= 0 or params['risk'] > 0.1:
            return False, "Risk must be between 0 and 10%"
        
        return True, "All parameters valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def has_results() -> bool:
    """Check if backtest results are available."""
    return (st.session_state.last_returns is not None and 
            st.session_state.last_trades is not None and 
            st.session_state.last_metrics is not None)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()