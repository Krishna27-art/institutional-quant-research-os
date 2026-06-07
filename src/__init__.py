"""
Root of the src package.
Sets up backward-compatible namespace aliases for renamed components.
"""

import sys

# Inject legacy names into sys.modules to prevent legacy imports from breaking
try:
    import src.alpha
    import src.alpha.prediction_registry
    import src.alpha.alphas
    import src.alpha.evolution
    import src.alpha.registry
    import src.alpha.ranker
    import src.alpha.decay
    
    sys.modules['src.alpha_factory'] = src.alpha
    sys.modules['src.alpha_factory.prediction_registry'] = src.alpha.prediction_registry
    sys.modules['src.alpha_factory.alphas'] = src.alpha.alphas
    sys.modules['src.alpha_factory.evolution'] = src.alpha.evolution
    sys.modules['src.alpha_factory.registry'] = src.alpha.registry
    sys.modules['src.alpha_factory.ranker'] = src.alpha.ranker
    sys.modules['src.alpha_factory.decay'] = src.alpha.decay
except ImportError as e:
    pass

try:
    import src.features
    import src.features.compute
    sys.modules['src.feature_store'] = src.features
    sys.modules['src.feature_store.compute'] = src.features.compute
except ImportError as e:
    pass

try:
    import src.regime
    sys.modules['src.regime_engine'] = src.regime
except ImportError as e:
    pass

try:
    import src.portfolio
    import src.portfolio.engine
    sys.modules['src.portfolio.allocator'] = src.portfolio.engine
except ImportError as e:
    pass

try:
    import src.risk
    import src.risk.institutional_risk_engine
    sys.modules['src.risk.risk_engine'] = src.risk.institutional_risk_engine
except ImportError as e:
    pass
