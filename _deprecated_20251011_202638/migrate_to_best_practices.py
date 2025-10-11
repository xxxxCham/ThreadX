#!/usr/bin/env python3
"""
Script de Migration vers Bonnes Pratiques ThreadX
==================================================

Applique progressivement les bonnes pratiques d'architecture au projet ThreadX existant.

Usage:
    python migrate_to_best_practices.py --phase 1
    python migrate_to_best_practices.py --phase 2 --dry-run
    python migrate_to_best_practices.py --all --force

Phases:
    1. Fondations (pyproject.toml, pre-commit, CI/CD)
    2. Architecture (Settings, DI, logging)
    3. Qualité (tests, docs, performance)

Auteur: ThreadX Core Team
Date: 11 octobre 2025
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Couleurs pour output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_header(text: str):
    """Affiche un header coloré."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{Colors.RESET}\n")


def print_success(text: str):
    """Message de succès."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_warning(text: str):
    """Message d'avertissement."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_error(text: str):
    """Message d'erreur."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def run_command(cmd: List[str], check: bool = True) -> Optional[str]:
    """Exécute une commande shell."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"Erreur commande: {' '.join(cmd)}")
        print_error(e.stderr)
        return None


class BestPracticesMigrator:
    """Gestionnaire de migration vers bonnes pratiques."""

    def __init__(self, project_root: Path, dry_run: bool = False):
        self.root = project_root
        self.dry_run = dry_run
        self.backup_dir = project_root / ".migration_backup"

    def create_backup(self):
        """Crée un backup avant migration."""
        if self.dry_run:
            print_warning("Mode dry-run: Pas de backup créé")
            return

        print_header("Création Backup")

        if self.backup_dir.exists():
            print_warning(f"Backup existant trouvé: {self.backup_dir}")
            response = input("Écraser? (y/N): ")
            if response.lower() != "y":
                print_error("Migration annulée")
                sys.exit(1)
            shutil.rmtree(self.backup_dir)

        self.backup_dir.mkdir()

        # Fichiers critiques à backup
        critical_files = [
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            ".gitignore",
        ]

        for file in critical_files:
            src = self.root / file
            if src.exists():
                shutil.copy2(src, self.backup_dir / file)
                print_success(f"Backup: {file}")

        print_success(f"Backup créé: {self.backup_dir}")

    # =========================================================
    #  PHASE 1: FONDATIONS
    # =========================================================

    def phase1_foundations(self):
        """Phase 1: Fondations (pyproject.toml, pre-commit, CI/CD)."""
        print_header("PHASE 1: FONDATIONS")

        self.create_pyproject_toml()
        self.create_precommit_config()
        self.create_github_workflows()
        self.create_gitignore()
        self.create_makefile()

        print_success("Phase 1 terminée!")

    def create_pyproject_toml(self):
        """Crée pyproject.toml moderne."""
        print("\n📄 Création pyproject.toml...")

        content = """[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "threadx"
version = "2.0.0"
description = "Framework de trading crypto haute performance"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "ThreadX Team", email = "team@threadx.dev"}
]

dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
    "toml>=0.10.2",
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "black>=23.12.0",
    "isort>=5.13.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[project.scripts]
threadx = "threadx.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--strict-markers", "--cov=src/threadx"]
"""

        if not self.dry_run:
            (self.root / "pyproject.toml").write_text(content, encoding="utf-8")

        print_success("pyproject.toml créé")

    def create_precommit_config(self):
        """Crée .pre-commit-config.yaml."""
        print("\n🔧 Création pre-commit config...")

        content = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
  
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.11
    hooks:
      - id: ruff
        args: [--fix]
"""

        if not self.dry_run:
            (self.root / ".pre-commit-config.yaml").write_text(content)

            # Installer pre-commit
            print("Installation pre-commit hooks...")
            run_command(["pip", "install", "pre-commit"])
            run_command(["pre-commit", "install"])

        print_success("Pre-commit configuré")

    def create_github_workflows(self):
        """Crée GitHub Actions workflows."""
        print("\n🔄 Création GitHub workflows...")

        workflows_dir = self.root / ".github" / "workflows"

        if not self.dry_run:
            workflows_dir.mkdir(parents=True, exist_ok=True)

        # CI workflow
        ci_content = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Lint
        run: |
          ruff check src/
          black --check src/
      
      - name: Test
        run: |
          pytest tests/ --cov=src/threadx --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
"""

        if not self.dry_run:
            (workflows_dir / "ci.yml").write_text(ci_content)

        print_success("GitHub workflows créés")

    def create_gitignore(self):
        """Crée/met à jour .gitignore."""
        print("\n📝 Mise à jour .gitignore...")

        gitignore_additions = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# MyPy
.mypy_cache/
.dmypy.json

# Ruff
.ruff_cache/

# Data (ne pas commit données)
data/
*.parquet
*.json
!configs/*.json

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Migration backup
.migration_backup/
"""

        gitignore_path = self.root / ".gitignore"

        if not self.dry_run:
            if gitignore_path.exists():
                current = gitignore_path.read_text()
                if gitignore_additions not in current:
                    gitignore_path.write_text(current + gitignore_additions)
            else:
                gitignore_path.write_text(gitignore_additions)

        print_success(".gitignore mis à jour")

    def create_makefile(self):
        """Crée Makefile avec commandes utiles."""
        print("\n🛠️  Création Makefile...")

        content = """
.PHONY: help install test lint format clean

help:
\t@echo "Commandes disponibles:"
\t@echo "  make install  - Installe dépendances dev"
\t@echo "  make test     - Lance les tests"
\t@echo "  make lint     - Vérifie qualité code"
\t@echo "  make format   - Formate le code"
\t@echo "  make clean    - Nettoie fichiers temp"

install:
\tpip install -e ".[dev]"
\tpre-commit install

test:
\tpytest tests/ --cov=src/threadx

lint:
\truff check src/
\tblack --check src/
\tisort --check-only src/

format:
\tblack src/
\tisort src/
\truff check src/ --fix

clean:
\tfind . -type d -name "__pycache__" -exec rm -rf {} +
\tfind . -type d -name "*.egg-info" -exec rm -rf {} +
\trm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
"""

        if not self.dry_run:
            (self.root / "Makefile").write_text(content)

        print_success("Makefile créé")

    # =========================================================
    #  PHASE 2: ARCHITECTURE
    # =========================================================

    def phase2_architecture(self):
        """Phase 2: Architecture (Settings, DI, logging)."""
        print_header("PHASE 2: ARCHITECTURE")

        self.create_settings_module()
        self.create_logging_utils()
        self.create_configs_directory()

        print_success("Phase 2 terminée!")

    def create_settings_module(self):
        """Crée module de configuration avec Pydantic."""
        print("\n⚙️  Création module Settings...")

        config_dir = self.root / "src" / "threadx" / "config"

        if not self.dry_run:
            config_dir.mkdir(parents=True, exist_ok=True)

        settings_content = '''"""Configuration centralisée ThreadX avec Pydantic."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field

class PathsConfig(BaseSettings):
    """Configuration des chemins."""
    json_root: Path = Field(default=Path("data/raw/json"))
    parquet_root: Path = Field(default=Path("data/processed/parquet"))
    indicators_db: Path = Field(default=Path("data/indicators"))
    
    class Config:
        env_prefix = "THREADX_"

class DataConfig(BaseSettings):
    """Configuration données."""
    history_days: int = Field(default=365, ge=1, le=3650)
    binance_limit: int = Field(default=1000, ge=1, le=1000)
    supported_timeframes: List[str] = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

class Settings(BaseSettings):
    """Configuration globale ThreadX."""
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    
    @classmethod
    def load_from_toml(cls, config_file: Path) -> "Settings":
        """Charge depuis fichier TOML."""
        import toml
        config = toml.load(config_file)
        return cls(**config)

def get_settings() -> Settings:
    """Récupère settings (singleton)."""
    return Settings()
'''

        if not self.dry_run:
            (config_dir / "settings.py").write_text(settings_content)
            (config_dir / "__init__.py").write_text(
                "from .settings import Settings, get_settings\n"
            )

        print_success("Module Settings créé")

    def create_logging_utils(self):
        """Crée utilitaires de logging."""
        print("\n📝 Création logging utils...")

        utils_dir = self.root / "src" / "threadx" / "utils"

        if not self.dry_run:
            utils_dir.mkdir(parents=True, exist_ok=True)

        logging_content = '''"""Configuration logging ThreadX."""

import logging
import logging.config
from pathlib import Path
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Configure le système de logging."""
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        }
    }
    
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "standard"
        }
        config["root"]["handlers"].append("file")
    
    logging.config.dictConfig(config)
    return logging.getLogger("threadx")

def get_logger(name: str) -> logging.Logger:
    """Récupère un logger nommé."""
    return logging.getLogger(name)
'''

        if not self.dry_run:
            (utils_dir / "logging_utils.py").write_text(logging_content)
            (utils_dir / "__init__.py").write_text("")

        print_success("Logging utils créé")

    def create_configs_directory(self):
        """Crée répertoire configs/ avec fichiers TOML."""
        print("\n📂 Création configs/...")

        configs_dir = self.root / "configs"

        if not self.dry_run:
            configs_dir.mkdir(exist_ok=True)

        default_config = """[paths]
json_root = "data/raw/json"
parquet_root = "data/processed/parquet"
indicators_db = "data/indicators"

[data]
history_days = 365
binance_limit = 1000
supported_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]

[indicators]
default_rsi_period = 14
default_bb_period = 20
default_bb_std = 2.0
"""

        if not self.dry_run:
            (configs_dir / "default.toml").write_text(default_config)

        print_success("Configs créés")

    # =========================================================
    #  PHASE 3: QUALITÉ
    # =========================================================

    def phase3_quality(self):
        """Phase 3: Qualité (tests, docs)."""
        print_header("PHASE 3: QUALITÉ")

        self.create_tests_structure()
        self.create_docs_structure()

        print_success("Phase 3 terminée!")

    def create_tests_structure(self):
        """Crée structure tests/."""
        print("\n🧪 Création structure tests...")

        tests_dir = self.root / "tests"

        if not self.dry_run:
            tests_dir.mkdir(exist_ok=True)
            (tests_dir / "unit").mkdir(exist_ok=True)
            (tests_dir / "integration").mkdir(exist_ok=True)
            (tests_dir / "fixtures").mkdir(exist_ok=True)

        conftest_content = '''"""Fixtures pytest globales."""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """DataFrame OHLCV de test."""
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    
    return pd.DataFrame({
        'open': close,
        'high': close + 50,
        'low': close - 50,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
'''

        if not self.dry_run:
            (tests_dir / "conftest.py").write_text(conftest_content)
            (tests_dir / "__init__.py").write_text("")

        print_success("Structure tests créée")

    def create_docs_structure(self):
        """Crée structure docs/."""
        print("\n📚 Création structure docs...")

        docs_dir = self.root / "docs"

        if not self.dry_run:
            docs_dir.mkdir(exist_ok=True)
            (docs_dir / "api").mkdir(exist_ok=True)
            (docs_dir / "tutorials").mkdir(exist_ok=True)

        index_content = """# ThreadX Documentation

Framework de trading crypto haute performance.

## Démarrage Rapide

```python
from threadx.config import get_settings
from threadx.indicators.numpy import rsi_np

settings = get_settings()
# ...
```

## Table des Matières

- [API Reference](api/)
- [Tutorials](tutorials/)
- [Architecture](BONNES_PRATIQUES_ARCHITECTURE.md)
"""

        if not self.dry_run:
            (docs_dir / "index.md").write_text(index_content)

        print_success("Structure docs créée")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Migration ThreadX vers bonnes pratiques"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Phase à exécuter (1: Fondations, 2: Architecture, 3: Qualité)",
    )
    parser.add_argument("--all", action="store_true", help="Exécuter toutes les phases")
    parser.add_argument(
        "--dry-run", action="store_true", help="Simulation sans modifications"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force l'exécution sans confirmation"
    )

    args = parser.parse_args()

    # Détection racine projet
    project_root = Path(__file__).parent

    print_header("MIGRATION THREADX - BONNES PRATIQUES")
    print(f"Projet: {project_root}")

    if args.dry_run:
        print_warning("MODE DRY-RUN: Aucune modification ne sera effectuée")

    if not args.force and not args.dry_run:
        print_warning("Cette migration va modifier votre projet!")
        response = input("Continuer? (y/N): ")
        if response.lower() != "y":
            print_error("Migration annulée")
            sys.exit(0)

    migrator = BestPracticesMigrator(project_root, dry_run=args.dry_run)

    # Backup
    if not args.dry_run:
        migrator.create_backup()

    # Exécution phases
    if args.all:
        migrator.phase1_foundations()
        migrator.phase2_architecture()
        migrator.phase3_quality()
    elif args.phase == 1:
        migrator.phase1_foundations()
    elif args.phase == 2:
        migrator.phase2_architecture()
    elif args.phase == 3:
        migrator.phase3_quality()
    else:
        parser.print_help()
        sys.exit(1)

    print_header("MIGRATION TERMINÉE")

    if not args.dry_run:
        print_success(f"Backup sauvegardé dans: {migrator.backup_dir}")
        print("\n📋 Prochaines étapes:")
        print("1. make install  # Installer dépendances")
        print("2. make test     # Lancer les tests")
        print("3. make lint     # Vérifier qualité code")
    else:
        print_warning("Mode dry-run: Relancer sans --dry-run pour appliquer")


if __name__ == "__main__":
    main()
