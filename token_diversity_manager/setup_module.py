#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration et Setup - TradXPro Token Diversity Manager
=========================================================

Script de configuration pour le module Token Diversity Manager.

Usage:
    python setup_module.py

Auteur: TradXPro Team
Date: 2 octobre 2025
"""

import os
import sys
from pathlib import Path

def setup_module():
    """Configure le module Token Diversity Manager"""
    
    print("🔧 SETUP - TRADXPRO TOKEN DIVERSITY MANAGER")
    print("=" * 50)
    
    # Vérification de l'environnement
    module_path = Path(__file__).parent
    print(f"📁 Chemin du module: {module_path}")
    
    # Vérification des fichiers
    required_files = [
        "tradxpro_core_manager.py",
        "__init__.py",
        "README.md"
    ]
    
    print("\n📋 Vérification des fichiers...")
    missing_files = []
    
    for file in required_files:
        file_path = module_path / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MANQUANT")
            missing_files.append(file)
    
    # Vérification des dossiers
    required_dirs = ["tests", "examples", "docs"]
    print(f"\n📂 Vérification des dossiers...")
    
    for dir_name in required_dirs:
        dir_path = module_path / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"✅ {dir_name}/ ({file_count} fichiers)")
        else:
            print(f"❌ {dir_name}/ - MANQUANT")
            missing_files.append(f"{dir_name}/")
    
    # Test d'importation
    print(f"\n🧪 Test d'importation...")
    try:
        sys.path.append(str(module_path))
        from tradxpro_core_manager import TradXProManager
        print("✅ Module importé avec succès")
        
        # Test basique
        manager = TradXProManager()
        print("✅ Gestionnaire initialisé")
        
    except Exception as e:
        print(f"❌ Erreur d'importation: {e}")
        return False
    
    # Résultat final
    print(f"\n📊 RÉSULTAT DU SETUP")
    print("-" * 30)
    
    if not missing_files:
        print("🎉 SETUP RÉUSSI !")
        print("✅ Tous les fichiers sont présents")
        print("✅ Module fonctionnel")
        print(f"\n🚀 Le module est prêt à être utilisé !")
        print(f"📁 Chemin: {module_path}")
        
        # Instructions d'utilisation
        print(f"\n💡 UTILISATION:")
        print(f"cd {module_path}")
        print("python tests/test_diversite_simple.py")
        print("python examples/quick_start_tradxpro.py")
        
        return True
    else:
        print("⚠️ SETUP INCOMPLET")
        print(f"❌ {len(missing_files)} fichiers manquants:")
        for file in missing_files:
            print(f"   • {file}")
        return False

def create_launcher_script():
    """Crée un script de lancement rapide"""
    
    module_path = Path(__file__).parent
    launcher_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lanceur Rapide - TradXPro Token Diversity Manager
================================================

Script de lancement rapide pour tester le module.
"""

import sys
from pathlib import Path

# Ajout du chemin du module
module_path = Path(__file__).parent
sys.path.append(str(module_path))

# Import et test
from tradxpro_core_manager import TradXProManager

def quick_test():
    """Test rapide du module"""
    print("🚀 TEST RAPIDE - TOKEN DIVERSITY MANAGER")
    print("=" * 50)
    
    try:
        # Initialisation
        manager = TradXProManager()
        print("✅ Gestionnaire initialisé")
        
        # Info du module
        from . import get_module_info, print_module_info
        print_module_info()
        
        print("\\n🎉 Module opérationnel !")
        
    except Exception as e:
        print(f"❌ Erreur: {{e}}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()
'''
    
    launcher_path = module_path / "launch_quick_test.py"
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(launcher_content)
    
    print(f"📝 Script de lancement créé: {launcher_path}")

def main():
    """Fonction principale"""
    
    try:
        # Setup principal
        success = setup_module()
        
        if success:
            # Créer le script de lancement
            create_launcher_script()
            
            print(f"\n" + "=" * 50)
            print("🎯 MODULE TOKEN DIVERSITY MANAGER CONFIGURÉ !")
            print("=" * 50)
            print("📦 Fonctionnalités disponibles:")
            print("   🔒 Diversité garantie des tokens")
            print("   📊 Analyse et rapports de diversité") 
            print("   📈 Indicateurs techniques intégrés")
            print("   ⚡ Téléchargements et cache optimisés")
            print("   🛠️ API simple et unifiée")
            
        else:
            print(f"\n⚠️ Configuration incomplète - Vérifiez les fichiers manquants")
            
    except KeyboardInterrupt:
        print(f"\n👋 Setup interrompu")
    except Exception as e:
        print(f"\n❌ Erreur setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()