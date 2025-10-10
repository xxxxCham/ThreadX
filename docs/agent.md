Règles et mode opératoire Codex (session 10.0.10)


Tu ecrire une courte phrase dans le fichier ""D:\ThreadX\docs\sessions\session-10.0.10.md", a chaque itération, pour conclure sur les modifivcations éffectuer.
But unique : déboguer et stabiliser ThreadX sans toucher à l’interface (UI existante).

Environnement : Windows/PowerShell, Python 3.12, layout src/.

Branche dédiée : codex/session-10.0.10 (aucune action en dehors de cette branche).



1\) Périmètre et interdits



À faire : corriger erreurs d’import, config, logique (Data Manager, IndicatorBank, BacktestEngine), robustifier les scripts d’entrée et le chargement de config, fiabiliser indicateurs/backtests.



À ne pas faire :



Ne crée ni ne modifie aucune UI (Tkinter/Streamlit/…).



Ne réorganise pas l’arborescence publique ni l’API exposée, sauf bug critique justifié.



Ne télécharge pas de données temps réel (backtests/données locales uniquement).



Ne committe pas logs, caches, artefacts, fichiers volumineux.



2\) Politique Git (strict)



Travailler uniquement sur codex/session-10.0.10.



Interdits : pas de push sur main, pas de merge sur main, pas de force-push sur main.



Livraison : toujours via Pull Request (PR) depuis codex/session-10.0.10.



Validation : aucune fusion tant que l’auteur n’a pas approuvé la PR.



Format des messages de commit (Conventional Commits) :

fix(config): forbid absolute paths when validation enabled

refactor(imports): add src fallback for run\_\* scripts

test(indicators): add smoke test for enrich\_indicators()



3\) Journal de session obligatoire (fichier unique)



À chaque correction, ajouter une ligne dans un seul et même fichier :

docs/sessions/session-10.0.10.md



Format d’une entrée (une seule ligne, ≤ 140 caractères) :

\[YYYY-MM-DD HH:mm]\[module] bref résumé de la correction; tests: <ok/échec>; PR: #<id>



Exemples :

\[2025-10-09 21:05]\[config] validation chemins absolus + défauts sûrs; tests: ok; PR: #27

\[2025-10-09 21:22]\[indicators] ordre déterministe d’application; tests: ok; PR: #28



Si la correction implique plusieurs fichiers, une seule ligne résumée.



Contenu initial proposé pour le fichier docs/sessions/session-10.0.10.md :



\# Journal de session 10.0.10



> Une entrée par correction, format :  

> \[YYYY-MM-DD HH:mm]\[module] résumé; tests: <ok/échec>; PR: #<id>





4\) Commandes et exécution (références)



Lancer depuis la racine du dépôt.



Activation \& Python path :



\& .\\.venv\\Scripts\\Activate.ps1

$env:PYTHONPATH = "$PWD;$PWD\\src"

$env:DISABLE\_PANDERA\_IMPORT\_WARNING = "True"





Tests ciblés (doivent passer) :



.\\.venv\\Scripts\\python.exe -m pytest -k "config or data\_manager or indicators" -q





Vérification import indicateurs :



.\\.venv\\Scripts\\python.exe -c "import importlib; m=importlib.import\_module('threadx.indicators.engine'); print('OK:', hasattr(m,'enrich\_indicators'))"





Lancement Data Manager (vérif boot uniquement, ne pas modifier l’UI) :



.\\.venv\\Scripts\\python.exe .\\apps\\data\_manager\\run\_data\_manager.py



5\) Règles d’implémentation



Imports robustes en tête de tout script exécutable (ex. run\_\*.py) :



import sys, os

ROOT = os.path.dirname(os.path.abspath(\_\_file\_\_))

SRC = os.path.join(os.path.dirname(ROOT), "src")

if SRC not in sys.path:

&nbsp;   sys.path.insert(0, SRC)





Un seul moteur de calcul (pas de chemins parallèles concurrents).



Déterminisme : pas d’itération sur set pour l’ordre d’application des indicateurs ; fixer l’ordre.



Types numériques harmonisés (éviter float32 vs float64 mélangés sur des opérations sensibles).



Parallélisme : futures avec timeouts et gestion propre des exceptions ; pas de deadlocks.



I/O : toujours with open(...) ; pas de fuites de descripteurs.



Cache : TTL/eviction raisonnable ; pas de croissance illimitée.



Config : validation stricte, valeurs par défaut sûres ; pas de unsafe\_load.



6\) Check-list par correction (à coller dans la PR)



&nbsp;Travaillé sur codex/session-10.0.10 uniquement.



&nbsp;Pas de modifications d’UI.



&nbsp;Scripts d’entrée : imports robustes OK.



&nbsp;pytest -k "config or data\_manager or indicators" : OK.



&nbsp;Ajout d’UNE ligne dans docs/sessions/session-10.0.10.md.



&nbsp;Aucun artefact/log/cache en suivi Git.



&nbsp;Description claire du bug, de la cause et du correctif.



7\) .gitignore (rappel minimal)

/.archive/

/logs/

/artifacts/

\*\*/\_\_pycache\_\_/

.pytest\_cache/

\*.log



8\) Résumé opérationnel pour Codex



Créer/mettre à jour la branche codex/session-10.0.10.



Reproduire le bug, corriger sans toucher l’UI.



Valider : imports robustes, tests ciblés OK, démarrage Data Manager OK.



Journaliser : ajouter une ligne dans docs/sessions/session-10.0.10.md.



Ouvrir une PR depuis codex/session-10.0.10 ; attendre validation explicite avant tout merge.

