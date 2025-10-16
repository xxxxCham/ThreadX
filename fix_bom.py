#!/usr/bin/env python3
"""Script pour corriger le BOM dans test_udfi_contract.py"""

file_path = "tests/phase_a/test_udfi_contract.py"

# Lire le fichier
with open(file_path, "rb") as f:
    content = f.read()

# Vérifier et supprimer BOM
has_bom = content.startswith(b"\xef\xbb\xbf")
print(f"BOM UTF-8 détecté: {has_bom}")

if has_bom:
    # Supprimer BOM
    content_clean = content[3:]

    # Écrire fichier corrigé
    with open(file_path, "wb") as f:
        f.write(content_clean)

    print(f"✅ BOM supprimé! Fichier corrigé: {file_path}")
else:
    print(f"ℹ️ Aucun BOM détecté dans {file_path}")
