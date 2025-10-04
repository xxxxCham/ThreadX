"""
Tests ThreadX Data Registry Module - Phase 2
Tests complets pour scan datasets et checksums.
"""

import pytest
import tempfile
from pathlib import Path
import hashlib

# Import du module à tester
try:
    from threadx.data.registry import (
        dataset_exists, scan_symbols, scan_timeframes, quick_inventory,
        file_checksum, dataset_info, cleanup_empty_datasets,
        RegistryError, _get_data_root, _get_processed_root, _build_dataset_path
    )
    from threadx.data.synth import make_synth_ohlcv
    from threadx.data.io import write_frame
except ImportError as e:
    pytest.skip(f"Modules ThreadX non disponibles: {e}", allow_module_level=True)


class TestPathBuilding:
    """Tests construction chemins datasets."""
    
    def test_build_dataset_path_default(self):
        """Test construction chemin avec racine par défaut."""
        path = _build_dataset_path("BTCUSDC", "15m")
        
        assert isinstance(path, Path)
        assert path.name == "15m.parquet"
        assert "BTCUSDC" in str(path)
        assert "processed" in str(path)
    
    def test_build_dataset_path_custom_root(self):
        """Test construction chemin avec racine custom."""
        custom_root = Path("/custom/data/root")
        path = _build_dataset_path("ETHUSDC", "1h", root=custom_root)
        
        assert path.parts[0] == "/"
        assert "custom" in str(path)
        assert path.name == "1h.parquet"
        assert "ETHUSDC" in str(path)
    
    def test_get_processed_root(self):
        """Test obtention racine processed/."""
        processed_root = _get_processed_root()
        
        assert isinstance(processed_root, Path)
        assert processed_root.name == "processed"


class TestDatasetExists:
    """Tests vérification existence datasets."""
    
    def test_dataset_exists_true(self, tmp_path):
        """Test dataset existant et valide."""
        # Structure: processed/BTCUSDC/15m.parquet
        symbol_dir = tmp_path / "processed" / "BTCUSDC"
        symbol_dir.mkdir(parents=True)
        
        dataset_path = symbol_dir / "15m.parquet"
        
        # Création fichier avec contenu
        df_test = make_synth_ohlcv(n=100, seed=42)
        write_frame(df_test, dataset_path)
        
        # Test existence
        exists = dataset_exists("BTCUSDC", "15m", root=tmp_path)
        
        assert exists is True
    
    def test_dataset_exists_false_missing(self, tmp_path):
        """Test dataset inexistant."""
        exists = dataset_exists("MISSING", "1h", root=tmp_path)
        
        assert exists is False
    
    def test_dataset_exists_false_empty(self, tmp_path):
        """Test fichier vide (invalide)."""
        symbol_dir = tmp_path / "processed" / "EMPTYUSDC"
        symbol_dir.mkdir(parents=True)
        
        # Fichier vide
        empty_file = symbol_dir / "1m.parquet"
        empty_file.touch()
        
        exists = dataset_exists("EMPTYUSDC", "1m", root=tmp_path)
        
        assert exists is False  # Fichier vide = invalide
    
    def test_dataset_exists_permission_error(self, tmp_path):
        """Test gestion erreurs permission."""
        # Simulation erreur (dossier inexistant + nom invalide)
        exists = dataset_exists("INVALID/PATH", "1h", root="/nonexistent/path")
        
        assert exists is False  # Erreur = considéré comme inexistant


class TestScanSymbols:
    """Tests scan symboles disponibles."""
    
    def test_scan_symbols_multiple(self, tmp_path):
        """Test scan multiple symboles."""
        processed_dir = tmp_path / "processed"
        
        # Création dossiers symboles
        symbols_to_create = ["BTCUSDC", "ETHUSDC", "ADAUSDC", "SOLUSDC"]
        for symbol in symbols_to_create:
            (processed_dir / symbol).mkdir(parents=True)
        
        # Scan
        found_symbols = scan_symbols(root=tmp_path)
        
        # Vérifications
        assert len(found_symbols) == 4
        assert set(found_symbols) == set(symbols_to_create)
        assert found_symbols == sorted(found_symbols)  # Ordre alphabétique
    
    def test_scan_symbols_empty(self, tmp_path):
        """Test scan sans symboles."""
        # Dossier processed vide
        (tmp_path / "processed").mkdir()
        
        symbols = scan_symbols(root=tmp_path)
        
        assert symbols == []
    
    def test_scan_symbols_no_processed_dir(self, tmp_path):
        """Test scan sans dossier processed."""
        symbols = scan_symbols(root=tmp_path)
        
        assert symbols == []
    
    def test_scan_symbols_ignores_files(self, tmp_path):
        """Test scan ignore les fichiers (seulement dossiers)."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        
        # Création mix dossiers + fichiers
        (processed_dir / "BTCUSDC").mkdir()  # Dossier → symbole valide
        (processed_dir / "ETHUSDC").mkdir()  # Dossier → symbole valide
        (processed_dir / "somefile.txt").touch()  # Fichier → ignoré
        
        symbols = scan_symbols(root=tmp_path)
        
        assert len(symbols) == 2
        assert "BTCUSDC" in symbols
        assert "ETHUSDC" in symbols


class TestScanTimeframes:
    """Tests scan timeframes par symbole."""
    
    def test_scan_timeframes_multiple(self, tmp_path):
        """Test scan multiple timeframes."""
        symbol_dir = tmp_path / "processed" / "BTCUSDC"
        symbol_dir.mkdir(parents=True)
        
        # Création fichiers timeframes
        timeframes_to_create = ["1m", "5m", "15m", "1h", "4h", "1d"]
        for tf in timeframes_to_create:
            tf_file = symbol_dir / f"{tf}.parquet"
            
            # Fichier avec contenu minimal
            df_test = make_synth_ohlcv(n=10, seed=42)
            write_frame(df_test, tf_file)
        
        # Scan
        found_timeframes = scan_timeframes("BTCUSDC", root=tmp_path)
        
        # Vérifications
        assert len(found_timeframes) == 6
        assert set(found_timeframes) == set(timeframes_to_create)
        assert found_timeframes == sorted(found_timeframes)
    
    def test_scan_timeframes_empty_symbol(self, tmp_path):
        """Test scan symbole sans timeframes."""
        symbol_dir = tmp_path / "processed" / "EMPTYUSDC"
        symbol_dir.mkdir(parents=True)
        
        timeframes = scan_timeframes("EMPTYUSDC", root=tmp_path)
        
        assert timeframes == []
    
    def test_scan_timeframes_missing_symbol(self, tmp_path):
        """Test scan symbole inexistant."""
        timeframes = scan_timeframes("MISSING", root=tmp_path)
        
        assert timeframes == []
    
    def test_scan_timeframes_ignores_non_parquet(self, tmp_path):
        """Test scan ignore fichiers non-.parquet."""
        symbol_dir = tmp_path / "processed" / "TESTUSDC"
        symbol_dir.mkdir(parents=True)
        
        # Mix fichiers .parquet et autres
        df_test = make_synth_ohlcv(n=5, seed=42)
        write_frame(df_test, symbol_dir / "1m.parquet")  # Valid
        write_frame(df_test, symbol_dir / "5m.parquet")  # Valid
        (symbol_dir / "readme.txt").touch()  # Ignoré
        (symbol_dir / "data.json").touch()   # Ignoré
        
        timeframes = scan_timeframes("TESTUSDC", root=tmp_path)
        
        assert len(timeframes) == 2
        assert "1m" in timeframes
        assert "5m" in timeframes


class TestQuickInventory:
    """Tests inventaire rapide complet."""
    
    def test_quick_inventory_complete(self, tmp_path):
        """Test inventaire complet multi-symboles."""
        processed_dir = tmp_path / "processed"
        
        # Structure complexe
        inventory_data = {
            "BTCUSDC": ["1m", "5m", "15m", "1h"],
            "ETHUSDC": ["1m", "15m", "4h"],
            "ADAUSDC": ["5m", "1h"],
            "SOLUSDC": ["1m"]
        }
        
        # Création structure
        for symbol, timeframes in inventory_data.items():
            symbol_dir = processed_dir / symbol
            symbol_dir.mkdir(parents=True)
            
            for tf in timeframes:
                tf_file = symbol_dir / f"{tf}.parquet"
                df_test = make_synth_ohlcv(n=5, seed=hash(symbol+tf) % 1000)
                write_frame(df_test, tf_file)
        
        # Inventaire
        inventory = quick_inventory(root=tmp_path)
        
        # Vérifications
        assert len(inventory) == 4
        assert inventory == inventory_data
        
        # Ordre symboles alphabétique
        assert list(inventory.keys()) == sorted(inventory.keys())
    
    def test_quick_inventory_empty(self, tmp_path):
        """Test inventaire sur structure vide."""
        (tmp_path / "processed").mkdir()
        
        inventory = quick_inventory(root=tmp_path)
        
        assert inventory == {}
    
    def test_quick_inventory_partial_symbols(self, tmp_path):
        """Test inventaire avec symboles partiels (certains sans timeframes)."""
        processed_dir = tmp_path / "processed"
        
        # Symbole avec données
        valid_dir = processed_dir / "VALIDUSDC"
        valid_dir.mkdir(parents=True)
        df_test = make_synth_ohlcv(n=5, seed=42)
        write_frame(df_test, valid_dir / "1m.parquet")
        
        # Symbole sans données (dossier vide)
        empty_dir = processed_dir / "EMPTYUSDC"
        empty_dir.mkdir(parents=True)
        
        inventory = quick_inventory(root=tmp_path)
        
        # Seulement symboles avec données
        assert len(inventory) == 1
        assert "VALIDUSDC" in inventory
        assert "EMPTYUSDC" not in inventory
    
    def test_quick_inventory_performance_no_read(self, tmp_path):
        """Test inventaire ne lit pas le contenu des fichiers."""
        processed_dir = tmp_path / "processed"
        
        # Création fichiers volumineux (simulation)
        symbol_dir = processed_dir / "PERFUSDC"
        symbol_dir.mkdir(parents=True)
        
        # Fichiers "volumineux" simulés (juste taille)
        large_file = symbol_dir / "1m.parquet"
        
        # Création fichier avec contenu minimal mais flag taille
        df_small = make_synth_ohlcv(n=10, seed=42)
        write_frame(df_small, large_file)
        
        # Inventaire doit être rapide (pas de lecture Parquet)
        import time
        start_time = time.perf_counter()
        
        inventory = quick_inventory(root=tmp_path)
        
        elapsed = time.perf_counter() - start_time
        
        # Vérifications
        assert "PERFUSDC" in inventory
        assert "1m" in inventory["PERFUSDC"]
        assert elapsed < 1.0  # Rapide (<1s même sur gros datasets)


class TestFileChecksum:
    """Tests calcul checksums."""
    
    def test_checksum_md5_default(self, tmp_path):
        """Test checksum MD5 par défaut."""
        test_file = tmp_path / "test.txt"
        test_content = b"Hello ThreadX Data Registry!"
        test_file.write_bytes(test_content)
        
        # Checksum via module
        checksum = file_checksum(test_file)
        
        # Vérification vs hashlib direct
        expected = hashlib.md5(test_content).hexdigest()
        assert checksum == expected
        assert len(checksum) == 32  # MD5 = 32 hex chars
    
    def test_checksum_sha256(self, tmp_path):
        """Test checksum SHA256."""
        test_file = tmp_path / "test.dat"
        test_content = b"SHA256 test content for ThreadX"
        test_file.write_bytes(test_content)
        
        checksum = file_checksum(test_file, algo="sha256")
        
        expected = hashlib.sha256(test_content).hexdigest()
        assert checksum == expected
        assert len(checksum) == 64  # SHA256 = 64 hex chars
    
    def test_checksum_large_file_streaming(self, tmp_path):
        """Test checksum streamé sur fichier volumineux."""
        large_file = tmp_path / "large.bin"
        
        # Simulation fichier 10MB avec pattern répétitif
        chunk_data = b"A" * 1024  # 1KB chunk
        with open(large_file, "wb") as f:
            for _ in range(10 * 1024):  # 10MB total
                f.write(chunk_data)
        
        # Checksum streamé
        checksum = file_checksum(large_file, chunk_size=4096)  # 4KB chunks
        
        # Vérification résultat cohérent
        assert len(checksum) == 32  # MD5
        assert checksum.isalnum()
    
    def test_checksum_missing_file(self, tmp_path):
        """Test erreur fichier manquant."""
        missing_file = tmp_path / "missing.txt"
        
        with pytest.raises(RegistryError, match="introuvable"):
            file_checksum(missing_file)
    
    def test_checksum_invalid_algorithm(self, tmp_path):
        """Test erreur algorithme invalide."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        with pytest.raises(RegistryError, match="invalide"):
            file_checksum(test_file, algo="invalid_algo")
    
    def test_checksum_directory_error(self, tmp_path):
        """Test erreur sur dossier (pas fichier)."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()
        
        with pytest.raises(RegistryError, match="pas un fichier"):
            file_checksum(test_dir)


class TestDatasetInfo:
    """Tests informations détaillées datasets."""
    
    def test_dataset_info_complete(self, tmp_path):
        """Test infos complètes dataset valide."""
        # Création dataset
        symbol_dir = tmp_path / "processed" / "INFOUSDC"
        symbol_dir.mkdir(parents=True)
        
        dataset_path = symbol_dir / "1h.parquet"
        df_test = make_synth_ohlcv(n=100, seed=42)
        write_frame(df_test, dataset_path)
        
        # Infos dataset
        info = dataset_info("INFOUSDC", "1h", root=tmp_path)
        
        # Vérifications
        assert info is not None
        assert info["symbol"] == "INFOUSDC"
        assert info["timeframe"] == "1h"
        assert info["size_bytes"] > 0
        assert info["size_mb"] > 0
        assert "checksum_md5" in info
        assert len(info["checksum_md5"]) == 32
    
    def test_dataset_info_missing(self, tmp_path):
        """Test infos dataset inexistant."""
        info = dataset_info("MISSING", "1h", root=tmp_path)
        
        assert info is None


class TestCleanupEmptyDatasets:
    """Tests nettoyage fichiers vides."""
    
    def test_cleanup_dry_run(self, tmp_path):
        """Test nettoyage en mode dry run."""
        processed_dir = tmp_path / "processed"
        
        # Création mix fichiers valides/vides
        valid_dir = processed_dir / "VALIDUSDC"
        valid_dir.mkdir(parents=True)
        
        # Fichier valide
        df_test = make_synth_ohlcv(n=10, seed=42)
        write_frame(df_test, valid_dir / "1m.parquet")
        
        # Fichiers vides
        (valid_dir / "empty1.parquet").touch()
        (valid_dir / "empty2.parquet").touch()
        
        # Nettoyage dry run
        stats = cleanup_empty_datasets(root=tmp_path, dry_run=True)
        
        # Vérifications
        assert stats["found"] == 2
        assert stats["removed"] == 0  # Pas de suppression en dry run
        
        # Fichiers toujours présents
        assert (valid_dir / "empty1.parquet").exists()
        assert (valid_dir / "empty2.parquet").exists()
    
    def test_cleanup_actual_removal(self, tmp_path):
        """Test nettoyage avec suppression réelle."""
        processed_dir = tmp_path / "processed"
        
        # Création fichiers vides
        test_dir = processed_dir / "CLEANUSDC"
        test_dir.mkdir(parents=True)
        
        empty_files = [
            test_dir / "empty1.parquet",
            test_dir / "empty2.parquet", 
            test_dir / "empty3.parquet"
        ]
        
        for empty_file in empty_files:
            empty_file.touch()
        
        # Suppression réelle
        stats = cleanup_empty_datasets(root=tmp_path, dry_run=False)
        
        # Vérifications
        assert stats["found"] == 3
        assert stats["removed"] == 3
        
        # Fichiers supprimés
        for empty_file in empty_files:
            assert not empty_file.exists()
    
    def test_cleanup_no_empty_files(self, tmp_path):
        """Test nettoyage sans fichiers vides."""
        processed_dir = tmp_path / "processed"
        
        # Création fichier valide seulement
        valid_dir = processed_dir / "NOEMPTYUSDC"
        valid_dir.mkdir(parents=True)
        
        df_test = make_synth_ohlcv(n=5, seed=42)
        write_frame(df_test, valid_dir / "1m.parquet")
        
        stats = cleanup_empty_datasets(root=tmp_path, dry_run=False)
        
        assert stats["found"] == 0
        assert stats["removed"] == 0