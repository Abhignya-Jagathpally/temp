"""
Clinical analysis modules for ResistanceMap.

Provides treatment response prediction, relapse detection, and outcome forecasting
across multiple myeloma patient cohorts.
"""

# Lazy imports with fallback handling
__all__ = [
    'treatment_conditioned',
    'dynamic_treatment_regimes',
    'relapse_prediction',
    'early_relapse_detection',
    'smm_progression',
    'mrd_prediction',
    'emd_prediction',
    'immunotherapy_response',
    'utils',
]


def __getattr__(name):
    """Lazy import clinical modules with error handling."""
    try:
        if name == 'treatment_conditioned':
            from . import treatment_conditioned
            return treatment_conditioned
        elif name == 'dynamic_treatment_regimes':
            from . import dynamic_treatment_regimes
            return dynamic_treatment_regimes
        elif name == 'relapse_prediction':
            from . import relapse_prediction
            return relapse_prediction
        elif name == 'early_relapse_detection':
            from . import early_relapse_detection
            return early_relapse_detection
        elif name == 'smm_progression':
            from . import smm_progression
            return smm_progression
        elif name == 'mrd_prediction':
            from . import mrd_prediction
            return mrd_prediction
        elif name == 'emd_prediction':
            from . import emd_prediction
            return emd_prediction
        elif name == 'immunotherapy_response':
            from . import immunotherapy_response
            return immunotherapy_response
        elif name == 'utils':
            from . import utils
            return utils
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    except ImportError as e:
        raise ImportError(f"Failed to import {name} from {__name__}: {str(e)}")


def __dir__():
    """List available clinical modules."""
    return sorted(__all__)
