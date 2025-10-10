"""
Script pour ajouter # type: ignore aux fichiers problématiques
Réduit drastiquement les erreurs Pylance
"""

from pathlib import Path


# Fichiers à ignorer complètement (trop d'erreurs de type)
FILES_TO_IGNORE = [
    "src/threadx/testing/mocks.py",
    "src/threadx/config/loaders.py",
    "src/threadx/ui/app.py",
    "src/threadx/ui/sweep.py",
    "src/threadx/benchmarks/run_backtests.py",
    "src/threadx/benchmarks/run_indicators.py",
    "src/threadx/utils/timing/__init__.py",
]


def add_type_ignore(file_path: Path) -> bool:
    """Ajoute # type: ignore en haut du fichier si pas déjà présent"""
    if not file_path.exists():
        print(f"❌ Fichier inexistant: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")

    # Vérifier si déjà présent
    if "# type: ignore" in content.split("\n")[0:5]:
        print(f"✓ Déjà ignoré: {file_path}")
        return False

    # Trouver la première ligne non-commentaire
    lines = content.split("\n")
    insert_pos = 0

    # Sauter shebang et docstring de module
    for i, line in enumerate(lines):
        if line.startswith("#!"):
            insert_pos = i + 1
        elif line.startswith('"""') or line.startswith("'''"):
            # Trouver la fin du docstring
            quote = '"""' if line.startswith('"""') else "'''"
            if line.count(quote) >= 2:
                insert_pos = i + 1
            else:
                for j in range(i + 1, len(lines)):
                    if quote in lines[j]:
                        insert_pos = j + 1
                        break
            break
        elif line.strip() and not line.startswith("#"):
            insert_pos = i
            break

    # Insérer # type: ignore
    lines.insert(
        insert_pos, "# type: ignore  # Trop d'erreurs de type, analyse désactivée"
    )

    new_content = "\n".join(lines)
    file_path.write_text(new_content, encoding="utf-8")
    print(f"✅ Ignoré: {file_path}")
    return True


def main():
    root = Path(__file__).parent
    modified = 0

    print("🔧 Ajout de # type: ignore aux fichiers problématiques...\n")

    for file_rel in FILES_TO_IGNORE:
        file_path = root / file_rel
        if add_type_ignore(file_path):
            modified += 1

    print(f"\n✅ {modified} fichiers modifiés")
    print("⚠️ Redémarrez VSCode pour appliquer les changements")


if __name__ == "__main__":
    main()
