from threadx.indicators.engine import enrich_indicators



# Configuration GPU (optionnel)

from threadx.utils.xp import gpu_available

print(f"GPU disponible: {gpu_available()}")