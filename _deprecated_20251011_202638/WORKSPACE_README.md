# 🎯 Workspace VS Code Unique - Guide Rapide

## ⚡ Démarrage Rapide

### 1️⃣ Ouvrir le workspace
```bash
# Méthode recommandée
code ThreadX.code-workspace

# Ou depuis VS Code
File > Open Workspace from File > ThreadX.code-workspace
```

### 2️⃣ Vérifier configuration
- VS Code devrait activer automatiquement `.venv`
- Terminal devrait afficher `(.venv)` au prompt
- Extensions recommandées devraient être proposées

### 3️⃣ Tester une tâche
```
Ctrl+Shift+P > Tasks: Run Task > ThreadX: Run All Tests
```

---

## 🧹 Nettoyage Configurations Multiples

### Exécuter le script de nettoyage
```powershell
# Dans PowerShell
.\cleanup_workspace.ps1
```

### Ce que fait le script
1. ✅ **Sauvegarde** `.vscode/settings.json` et `configs/pyrightconfig.json`
2. ✅ **Supprime** fichiers redondants consolidés dans le workspace
3. ✅ **Valide** que `ThreadX.code-workspace` est complet
4. ✅ **Archive** dans `.archive/workspace_backup_[date]/`

---

## 📋 Configurations Incluses

### Settings (Python, Format, Linting)
- Python: `.venv` auto-activé, PYTHONPATH inclut `src/`
- Format: Black (88 chars), format on save
- Type checking: Désactivé pour performance
- Linting: Désactivé (dev mode)

### Launch (6 configs debug)
- `F5` > Python: Fichier actuel
- `F5` > ThreadX: Update Daily Tokens
- `F5` > ThreadX: Analyze Token
- `F5` > ThreadX: Scan All Tokens
- `F5` > ThreadX: Tests (pytest)
- `F5` > ThreadX: Test End-to-End

### Tasks (6 tâches)
- `Ctrl+Shift+P` > Tasks > ThreadX: Update Daily Tokens
- `Ctrl+Shift+P` > Tasks > ThreadX: Analyze Token
- `Ctrl+Shift+P` > Tasks > ThreadX: Scan All Tokens
- `Ctrl+Shift+P` > Tasks > ThreadX: Run All Tests
- `Ctrl+Shift+P` > Tasks > ThreadX: Test End-to-End
- `Ctrl+Shift+P` > Tasks > ThreadX: Install Requirements

---

## 🔧 Personnalisation

### Modifier paramètres d'une tâche
1. Ouvrir `ThreadX.code-workspace`
2. Chercher section `"tasks"`
3. Modifier `"args"` de la tâche souhaitée
4. Sauvegarder (reload workspace si nécessaire)

**Exemple** - Changer nombre de tokens:
```json
{
    "label": "ThreadX: Update Daily Tokens",
    "args": [
        "scripts/update_daily_tokens.py",
        "--tokens", "150",  // ← Modifier ici (était 100)
        "--timeframes", "15m,1h,4h"  // ← Ajouter timeframes
    ]
}
```

### Ajouter une nouvelle tâche
```json
{
    "label": "Mon Script Custom",
    "type": "shell",
    "command": "${config:python.defaultInterpreterPath}",
    "args": ["mon_script.py", "--arg1", "value"],
    "presentation": {
        "reveal": "always",
        "panel": "new"
    }
}
```

---

## ❓ FAQ

### **Q: Je vois encore `.vscode/settings.json`, dois-je le supprimer ?**
A: Oui, il est redondant. Le workspace inclut déjà tous les settings. Utilisez `cleanup_workspace.ps1` ou supprimez manuellement.

### **Q: Quelle différence entre ouvrir le dossier vs le workspace ?**
A: 
- **Dossier** (`code d:\ThreadX`): VS Code charge workspace s'il existe
- **Workspace** (`code ThreadX.code-workspace`): Force utilisation workspace
- **Recommandé**: Toujours utiliser workspace pour garantir configs

### **Q: Puis-je avoir plusieurs workspaces pour ThreadX ?**
A: **Non recommandé**. Un workspace unique évite :
- Configurations contradictoires
- Settings dupliqués
- Confusion Python interpreter
- Problèmes PYTHONPATH

### **Q: Comment partager le workspace avec l'équipe ?**
A: 
1. Commiter `ThreadX.code-workspace` dans Git
2. Chaque membre clone repo et ouvre workspace
3. VS Code propose automatiquement installer extensions recommandées

### **Q: Le workspace contient des chemins absolus ?**
A: Non ! Le workspace utilise `${workspaceFolder}` (relatif). Fonctionne quel que soit le chemin d'installation.

---

## 📖 Documentation Complète

Voir **[docs/WORKSPACE_CONFIGURATION.md](docs/WORKSPACE_CONFIGURATION.md)** pour :
- Configuration détaillée
- Structure projet
- Utilisation quotidienne
- Conseils performance
- Troubleshooting

---

## ✅ Checklist Post-Nettoyage

- [ ] Exécuté `cleanup_workspace.ps1`
- [ ] Fermé VS Code
- [ ] Rouvert avec `code ThreadX.code-workspace`
- [ ] Vérifié `.venv` activé dans terminal
- [ ] Testé F5 (config debug fonctionne)
- [ ] Testé Tasks (tâche s'exécute)
- [ ] Extensions recommandées installées

---

**✨ Workspace ThreadX prêt pour développement productif !**
