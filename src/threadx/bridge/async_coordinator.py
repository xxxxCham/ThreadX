"""
ThreadX Async Coordinator - Non-blocking Bridge Orchestration
=============================================================

Orchestrateur asynchrone central pour ThreadX Bridge.
Délègue calculs lourds (backtest, indicators, sweep) à ThreadPoolExecutor
et fournit interface non-bloquante pour Dash UI + CLI.

Architecture:
    Dash UI → ThreadXBridge.run_*_async() → ThreadPoolExecutor worker
                ↓ (immédiat, non-bloquant)
           Queue + polling get_event()
                ↓
           Update UI (Graph, Tables)

Usage Dash (polling):
    >>> bridge = ThreadXBridge(max_workers=4)
    >>> req = BacktestRequest(symbol='BTCUSDT', ...)
    >>> future = bridge.run_backtest_async(req)
    >>> # Polling dans dcc.Interval callback
    >>> event = bridge.get_event(timeout=0.5)
    >>> if event and event['type'] == 'backtest_done':
    ...     result = event['payload']
    ...     update_graph(result.equity_curve)

Usage CLI (sync via Future):
    >>> bridge = ThreadXBridge()
    >>> req = BacktestRequest(symbol='BTCUSDT', ...)
    >>> future = bridge.run_backtest_async(req)
    >>> result = future.result(timeout=300)
    >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")

Thread-Safety:
    - active_tasks protégé par state_lock
    - Queue thread-safe nativement
    - Futures thread-safe nativement
    - Pas de data races

Author: ThreadX Framework
Version: Prompt 3 - Async Coordinator
"""

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, Callable, Dict

from threadx.bridge.controllers import (
    BacktestController,
    DataController,
    IndicatorController,
    SweepController,
)
from threadx.bridge.exceptions import BridgeError
from threadx.bridge.models import (
    BacktestRequest,
    BacktestResult,
    Configuration,
    DataRequest,
    DataValidationResult,
    IndicatorRequest,
    IndicatorResult,
    SweepRequest,
    SweepResult,
)

logger = logging.getLogger(__name__)


def _generate_task_id() -> str:
    """Génère UUID court (8 char) pour task tracking.

    Returns:
        Identifiant unique task (ex: 'a3f2c8b1').
    """
    return str(uuid.uuid4())[:8]


class ThreadXBridge:
    """Orchestrateur asynchrone ThreadX Bridge.

    Coordonne exécution non-bloquante de backtests, calculs indicateurs,
    parameter sweeps et validation données via ThreadPoolExecutor.

    Fournit:
        - Méthodes async pour soumettre tâches (run_*_async)
        - Queue thread-safe pour résultats
        - Polling non-bloquant (get_event)
        - Callbacks optionnels
        - State management (get_state, cancel_task)

    Attributes:
        executor: ThreadPoolExecutor pour délégation calculs.
        controllers: Dict des 4 controllers Bridge (backtest, etc.).
        results_queue: Queue thread-safe événements résultats.
        state_lock: Lock protection accès concurrent active_tasks.
        active_tasks: Dict {task_id: Future} tâches en cours.
        config: Configuration Bridge globale.

    Thread-Safety:
        Toutes mutations active_tasks sous state_lock.
        Queue nativement thread-safe.
        Futures nativement thread-safe.
    """

    def __init__(
        self,
        max_workers: int = 4,
        config: Configuration | None = None,
    ) -> None:
        """Initialise ThreadXBridge avec ThreadPoolExecutor et controllers.

        Args:
            max_workers: Nombre max workers parallèles (default: 4).
            config: Configuration Bridge ou None (utilise defaults).

        Example:
            >>> bridge = ThreadXBridge(max_workers=8)
            >>> state = bridge.get_state()
            >>> print(f"Workers: {state['max_workers']}")
        """
        self.config = config or Configuration(max_workers=max_workers)

        # ThreadPoolExecutor : délègue calculs lourds
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="threadx-worker-",
        )

        # Controllers Bridge (P2)
        self.controllers: Dict[str, Any] = {
            "backtest": BacktestController(self.config),
            "indicator": IndicatorController(self.config),
            "sweep": SweepController(self.config),
            "data": DataController(self.config),
        }

        # Queue thread-safe : événements résultats
        # Format: (event_type, task_id, payload)
        self.results_queue: Queue[tuple[str, str, Any]] = Queue()

        # Lock : protection active_tasks
        self.state_lock = threading.Lock()

        # Tâches actives : {task_id: Future}
        self.active_tasks: Dict[str, Future[Any]] = {}

        # Compteurs monitoring
        self._task_counter = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

        logger.info(
            f"ThreadXBridge initialized: max_workers={max_workers}, "
            f"config={self.config}"
        )

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API : Async Task Submission
    # ═══════════════════════════════════════════════════════════════

    def run_backtest_async(
        self,
        req: BacktestRequest,
        callback: (
            Callable[[BacktestResult | None, Exception | None], None] | None
        ) = None,
        task_id: str | None = None,
    ) -> Future[BacktestResult]:
        """Soumet backtest en thread worker (non-bloquant).

        Exécute BacktestController.run_backtest() en arrière-plan via
        ThreadPoolExecutor. Enqueue résultat pour polling get_event().

        Args:
            req: BacktestRequest validée avec symbol, strategy, params.
            callback: Fonction optionnelle appelée au résultat.
                     Signature: callback(result=...) ou callback(error=...)
            task_id: Identifiant unique (généré si None).

        Returns:
            Future pour attendre résultat (CLI) ou ignorer (Dash polling).

        Raises:
            BridgeError: Si executor fermé ou requête invalide.

        Example:
            >>> req = BacktestRequest(symbol='BTCUSDT', strategy='bb', ...)
            >>> future = bridge.run_backtest_async(req)
            >>> result = future.result(timeout=300)  # CLI
            >>> # OU
            >>> event = bridge.get_event(timeout=0.5)  # Dash polling

        Thread-Safety:
            Soumet task dans executor.
            Enregistre Future dans active_tasks sous lock.
        """
        if task_id is None:
            task_id = _generate_task_id()

        # Validation basique
        if self.config.validate_requests and not req.validate():
            raise BridgeError(
                f"Invalid BacktestRequest for task {task_id}: "
                f"missing required fields"
            )

        logger.info(
            f"Task {task_id} submitted: backtest "
            f"{req.symbol} {req.timeframe} {req.strategy}"
        )

        # Soumettre task en executor
        future = self.executor.submit(
            self._run_backtest_wrapped, req, callback, task_id
        )

        # Enregistrer task active (thread-safe)
        with self.state_lock:
            self.active_tasks[task_id] = future
            self._task_counter += 1

        return future

    def run_indicator_async(
        self,
        req: IndicatorRequest,
        callback: (
            Callable[[IndicatorResult | None, Exception | None], None] | None
        ) = None,
        task_id: str | None = None,
    ) -> Future[IndicatorResult]:
        """Soumet calcul indicateurs en thread worker (non-bloquant).

        Exécute IndicatorController.build_indicators() en arrière-plan.

        Args:
            req: IndicatorRequest avec symbol, indicators dict.
            callback: Fonction optionnelle appelée au résultat.
            task_id: Identifiant unique (généré si None).

        Returns:
            Future pour attendre résultat.

        Raises:
            BridgeError: Si executor fermé ou requête invalide.

        Example:
            >>> req = IndicatorRequest(
            ...     symbol='BTCUSDT',
            ...     timeframe='1h',
            ...     indicators={'ema': {'period': 50}}
            ... )
            >>> future = bridge.run_indicator_async(req)
        """
        if task_id is None:
            task_id = _generate_task_id()

        if self.config.validate_requests and not req.validate():
            raise BridgeError(f"Invalid IndicatorRequest for task {task_id}")

        logger.info(
            f"Task {task_id} submitted: indicators "
            f"{req.symbol} {req.timeframe} "
            f"{list(req.indicators.keys())}"
        )

        future = self.executor.submit(
            self._run_indicator_wrapped, req, callback, task_id
        )

        with self.state_lock:
            self.active_tasks[task_id] = future
            self._task_counter += 1

        return future

    def run_sweep_async(
        self,
        req: SweepRequest,
        callback: Callable[[SweepResult | None, Exception | None], None] | None = None,
        task_id: str | None = None,
    ) -> Future[SweepResult]:
        """Soumet parameter sweep en thread worker (non-bloquant).

        Exécute SweepController.run_sweep() en arrière-plan.

        Args:
            req: SweepRequest avec param_grid et critères.
            callback: Fonction optionnelle appelée au résultat.
            task_id: Identifiant unique (généré si None).

        Returns:
            Future pour attendre résultat.

        Raises:
            BridgeError: Si executor fermé ou requête invalide.

        Example:
            >>> req = SweepRequest(
            ...     symbol='BTCUSDT',
            ...     strategy='bb',
            ...     param_grid={'period': [10, 20, 30]}
            ... )
            >>> future = bridge.run_sweep_async(req)
        """
        if task_id is None:
            task_id = _generate_task_id()

        if self.config.validate_requests and not req.validate():
            raise BridgeError(f"Invalid SweepRequest for task {task_id}")

        logger.info(
            f"Task {task_id} submitted: sweep "
            f"{req.symbol} {req.strategy} "
            f"grid_size={len(req.param_grid)}"
        )

        future = self.executor.submit(self._run_sweep_wrapped, req, callback, task_id)

        with self.state_lock:
            self.active_tasks[task_id] = future
            self._task_counter += 1

        return future

    def validate_data_async(
        self,
        req: DataRequest,
        callback: (
            Callable[[DataValidationResult | None, Exception | None], None] | None
        ) = None,
        task_id: str | None = None,
    ) -> Future[DataValidationResult]:
        """Soumet validation données en thread worker (non-bloquant).

        Exécute DataController.validate_data() en arrière-plan.

        Args:
            req: DataRequest avec symbol, timeframe.
            callback: Fonction optionnelle appelée au résultat.
            task_id: Identifiant unique (généré si None).

        Returns:
            Future pour attendre résultat.

        Raises:
            BridgeError: Si executor fermé ou requête invalide.

        Example:
            >>> req = DataRequest(symbol='BTCUSDT', timeframe='1h')
            >>> future = bridge.validate_data_async(req)
            >>> result = future.result()
            >>> print(f"Quality: {result.quality_score}/10")
        """
        if task_id is None:
            task_id = _generate_task_id()

        if self.config.validate_requests and not req.validate_request():
            raise BridgeError(f"Invalid DataRequest for task {task_id}")

        logger.info(
            f"Task {task_id} submitted: data validation "
            f"{req.symbol} {req.timeframe}"
        )

        future = self.executor.submit(
            self._validate_data_wrapped, req, callback, task_id
        )

        with self.state_lock:
            self.active_tasks[task_id] = future
            self._task_counter += 1

        return future

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API : Polling & State Management
    # ═══════════════════════════════════════════════════════════════

    def get_event(self, timeout: float = 0.1) -> Dict[str, Any] | None:
        """Récupère événement résultat de queue (non-bloquant).

        Utilisé par Dash polling (dcc.Interval) pour récupérer résultats
        de tâches async sans bloquer UI thread.

        Args:
            timeout: Timeout queue.get() en secondes (0.1 par défaut).

        Returns:
            Dict avec clés:
                - 'type': 'backtest_done' | 'indicator_done' | 'sweep_done'
                         | 'data_validated' | 'error'
                - 'task_id': str identifiant tâche
                - 'payload': BacktestResult | IndicatorResult | str error
            ou None si queue vide.

        Example:
            >>> # Dans Dash dcc.Interval callback (500ms)
            >>> event = bridge.get_event(timeout=0.1)
            >>> if event:
            ...     if event['type'] == 'backtest_done':
            ...         result = event['payload']
            ...         fig = plot_equity(result.equity_curve)
            ...     elif event['type'] == 'error':
            ...         error_msg = event['payload']
            ...         display_error(error_msg)

        Thread-Safety:
            Queue.get() nativement thread-safe.
        """
        try:
            (event_type, task_id, payload) = self.results_queue.get(timeout=timeout)

            logger.debug(f"Event retrieved: {event_type} for task {task_id}")

            return {"type": event_type, "task_id": task_id, "payload": payload}

        except Empty:
            return None

    def get_state(self) -> Dict[str, Any]:
        """Retourne état global Bridge (thread-safe).

        Returns:
            Dict avec métriques:
                - 'active_tasks': int nombre tâches en cours
                - 'queue_size': int résultats en attente
                - 'max_workers': int workers configurés
                - 'total_submitted': int total tâches soumises
                - 'total_completed': int total tâches terminées
                - 'total_failed': int total tâches échouées
                - 'xp_layer': str backend calcul ('numpy' | 'cupy')

        Example:
            >>> state = bridge.get_state()
            >>> if state['queue_size'] > 0:
            ...     print(f"Résultats en attente: {state['queue_size']}")
            >>> print(f"Workers actifs: {state['active_tasks']}")

        Thread-Safety:
            Lecture active_tasks sous lock.
        """
        with self.state_lock:
            active_count = len(self.active_tasks)
            total_submitted = self._task_counter
            total_completed = self._completed_tasks
            total_failed = self._failed_tasks

        return {
            "active_tasks": active_count,
            "queue_size": self.results_queue.qsize(),
            "max_workers": self.config.max_workers,
            "total_submitted": total_submitted,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "xp_layer": self.config.xp_layer,
        }

    def cancel_task(self, task_id: str) -> bool:
        """Annule tâche active si pas encore exécutée.

        Args:
            task_id: Identifiant tâche à annuler.

        Returns:
            True si annulée avec succès, False si déjà exécutée/complétée.

        Example:
            >>> future = bridge.run_backtest_async(req)
            >>> task_id = 'abc123'
            >>> if bridge.cancel_task(task_id):
            ...     print("Backtest annulé")

        Thread-Safety:
            Accès active_tasks sous lock.
        """
        with self.state_lock:
            future = self.active_tasks.get(task_id)

            if future is None:
                logger.warning(f"Cannot cancel task {task_id}: not found")
                return False

            # Tenter annulation (fonctionne si pas encore started)
            cancelled = future.cancel()

            if cancelled:
                self.active_tasks.pop(task_id, None)
                logger.info(f"Task {task_id} cancelled successfully")
            else:
                logger.warning(f"Task {task_id} already running, cannot cancel")

            return cancelled

    # ═══════════════════════════════════════════════════════════════
    # PUBLIC API : Lifecycle
    # ═══════════════════════════════════════════════════════════════

    def shutdown(self, wait: bool = True, timeout: float | None = None):
        """Ferme ThreadPoolExecutor proprement.

        Args:
            wait: Attendre fin tâches avant retour (default: True).
            timeout: Timeout max attente en secondes (None = illimité).

        Example:
            >>> bridge.shutdown(wait=True, timeout=60)
            >>> # Toutes tâches terminées ou timeout

        Thread-Safety:
            executor.shutdown() thread-safe nativement.
        """
        logger.info(f"Shutting down ThreadXBridge " f"(wait={wait}, timeout={timeout})")

        self.executor.shutdown(wait=wait, cancel_futures=not wait)

        if wait and timeout:
            start = time.time()
            while self.results_queue.qsize() > 0:
                if time.time() - start > timeout:
                    logger.warning(
                        f"Shutdown timeout: {self.results_queue.qsize()} "
                        f"events still in queue"
                    )
                    break
                time.sleep(0.1)

        logger.info("ThreadXBridge shutdown complete")

    # ═══════════════════════════════════════════════════════════════
    # INTERNAL : Wrapped Workers (executed in ThreadPoolExecutor)
    # ═══════════════════════════════════════════════════════════════

    def _run_backtest_wrapped(
        self,
        req: BacktestRequest,
        callback: Callable[[BacktestResult | None, Exception | None], None] | None,
        task_id: str,
    ) -> BacktestResult:
        """Wrapper backtest exécuté dans thread worker.

        Workflow:
            1. Appelle BacktestController.run_backtest(req)
            2. Enqueue résultat en results_queue
            3. Exécute callback si fourni
            4. Gère exceptions → BridgeError en queue
            5. Cleanup active_tasks (finally)

        Args:
            req: BacktestRequest validée.
            callback: Callback optionnel.
            task_id: Identifiant task.

        Returns:
            BacktestResult pour Future.result().

        Raises:
            Re-raise exception après enqueue pour Future.

        Thread-Safety:
            Cleanup active_tasks sous lock.
        """
        start_time = time.time()

        try:
            logger.info(f"Task {task_id} executing: backtest started")

            # Appel controller (peut lever BacktestError)
            result = self.controllers["backtest"].run_backtest(req)

            exec_time = time.time() - start_time
            logger.info(
                f"Task {task_id} completed: backtest "
                f"(exec_time={exec_time:.2f}s, "
                f"sharpe={result.sharpe_ratio:.2f})"
            )

            # Enqueue succès
            self.results_queue.put(("backtest_done", task_id, result))

            # Callback succès
            if callback:
                try:
                    callback(result, None)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            # Compteur succès
            with self.state_lock:
                self._completed_tasks += 1

            return result

        except BridgeError as e:
            # Erreur attendue
            logger.error(f"Task {task_id} BridgeError: {e}")
            error_msg = f"BridgeError: {str(e)}"

            self.results_queue.put(("error", task_id, error_msg))

            if callback:
                try:
                    callback(None, e)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            with self.state_lock:
                self._failed_tasks += 1

            raise  # Re-raise pour Future

        except Exception as e:
            # Erreur non prévue
            logger.exception(f"Task {task_id} unexpected error")
            error_msg = f"InternalError: {str(e)}"

            self.results_queue.put(("error", task_id, error_msg))

            if callback:
                try:
                    callback(None, e)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            with self.state_lock:
                self._failed_tasks += 1

            raise  # Re-raise pour Future

        finally:
            # Cleanup active_tasks (thread-safe)
            with self.state_lock:
                self.active_tasks.pop(task_id, None)

    def _run_indicator_wrapped(
        self,
        req: IndicatorRequest,
        callback: Callable[[IndicatorResult | None, Exception | None], None] | None,
        task_id: str,
    ) -> IndicatorResult:
        """Wrapper indicateurs exécuté dans thread worker.

        Même pattern que _run_backtest_wrapped.
        """
        start_time = time.time()

        try:
            logger.info(f"Task {task_id} executing: indicators started")

            result = self.controllers["indicator"].build_indicators(req)

            exec_time = time.time() - start_time
            logger.info(
                f"Task {task_id} completed: indicators "
                f"(exec_time={exec_time:.2f}s, "
                f"cache_hits={result.cache_hits})"
            )

            self.results_queue.put(("indicator_done", task_id, result))

            if callback:
                try:
                    callback(result, None)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            with self.state_lock:
                self._completed_tasks += 1

            return result

        except Exception as e:
            logger.exception(f"Task {task_id} indicator error")
            error_msg = f"IndicatorError: {str(e)}"

            self.results_queue.put(("error", task_id, error_msg))

            if callback:
                try:
                    callback(None, e)
                except Exception:
                    pass

            with self.state_lock:
                self._failed_tasks += 1

            raise

        finally:
            with self.state_lock:
                self.active_tasks.pop(task_id, None)

    def _run_sweep_wrapped(
        self,
        req: SweepRequest,
        callback: Callable[[SweepResult | None, Exception | None], None] | None,
        task_id: str,
    ) -> SweepResult:
        """Wrapper sweep exécuté dans thread worker.

        Même pattern que _run_backtest_wrapped.
        """
        start_time = time.time()

        try:
            logger.info(f"Task {task_id} executing: sweep started")

            result = self.controllers["sweep"].run_sweep(req)

            exec_time = time.time() - start_time
            logger.info(
                f"Task {task_id} completed: sweep "
                f"(exec_time={exec_time:.2f}s, "
                f"best_sharpe={result.best_sharpe:.2f})"
            )

            self.results_queue.put(("sweep_done", task_id, result))

            if callback:
                try:
                    callback(result, None)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            with self.state_lock:
                self._completed_tasks += 1

            return result

        except Exception as e:
            logger.exception(f"Task {task_id} sweep error")
            error_msg = f"SweepError: {str(e)}"

            self.results_queue.put(("error", task_id, error_msg))

            if callback:
                try:
                    callback(None, e)
                except Exception:
                    pass

            with self.state_lock:
                self._failed_tasks += 1

            raise

        finally:
            with self.state_lock:
                self.active_tasks.pop(task_id, None)

    def _validate_data_wrapped(
        self,
        req: DataRequest,
        callback: (
            Callable[[DataValidationResult | None, Exception | None], None] | None
        ),
        task_id: str,
    ) -> DataValidationResult:
        """Wrapper validation données exécuté dans thread worker.

        Même pattern que _run_backtest_wrapped.
        """
        start_time = time.time()

        try:
            logger.info(f"Task {task_id} executing: data validation started")

            result = self.controllers["data"].validate_data(req)

            exec_time = time.time() - start_time
            logger.info(
                f"Task {task_id} completed: data validation "
                f"(exec_time={exec_time:.2f}s, "
                f"quality={result.quality_score:.1f}/10)"
            )

            self.results_queue.put(("data_validated", task_id, result))

            if callback:
                try:
                    callback(result, None)
                except Exception as cb_err:
                    logger.error(f"Task {task_id} callback error: {cb_err}")

            with self.state_lock:
                self._completed_tasks += 1

            return result

        except Exception as e:
            logger.exception(f"Task {task_id} data validation error")
            error_msg = f"DataError: {str(e)}"

            self.results_queue.put(("error", task_id, error_msg))

            if callback:
                try:
                    callback(None, e)
                except Exception:
                    pass

            with self.state_lock:
                self._failed_tasks += 1

            raise

        finally:
            with self.state_lock:
                self.active_tasks.pop(task_id, None)
