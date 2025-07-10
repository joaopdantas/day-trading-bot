# app/api/v1/indicators/__init__.py
"""
Módulo de indicadores técnicos para MakesALot Trading API
"""

from .technical import TechnicalIndicators

__all__ = ['TechnicalIndicators']

# app/api/v1/endpoints/__init__.py  
"""
Endpoints da API MakesALot Trading
"""

# Não importar aqui para evitar dependências circulares
# Os imports serão feitos diretamente nos arquivos que precisam

__all__ = []

# app/api/v1/__init__.py
"""
API v1 MakesALot Trading
"""

__all__ = []

# app/api/__init__.py
"""
Módulo API MakesALot Trading
"""

__all__ = []

# app/__init__.py
"""
MakesALot Trading Application
"""

__version__ = "2.0.0"
__description__ = "API avançada para análise técnica e previsões de trading"

__all__ = []