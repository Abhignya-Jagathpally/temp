"""
Infrastructure and data processing modules for ResistanceMap.

Handles data integration, quality control, harmonization, and advanced
uncertainty quantification across multimodal patient data.
"""

# Lazy imports with fallback handling
__all__ = [
    'missing_modality',
    'continual_learning',
    'uncertainty_quantification',
    'multi_site_harmonization',
    'scrna_integration',
    'imaging_radiomics',
]


def __getattr__(name):
    """Lazy import infrastructure modules with error handling."""
    try:
        if name == 'missing_modality':
            from . import missing_modality
            return missing_modality
        elif name == 'continual_learning':
            from . import continual_learning
            return continual_learning
        elif name == 'uncertainty_quantification':
            from . import uncertainty_quantification
            return uncertainty_quantification
        elif name == 'multi_site_harmonization':
            from . import multi_site_harmonization
            return multi_site_harmonization
        elif name == 'scrna_integration':
            from . import scrna_integration
            return scrna_integration
        elif name == 'imaging_radiomics':
            from . import imaging_radiomics
            return imaging_radiomics
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    except ImportError as e:
        raise ImportError(f"Failed to import {name} from {__name__}: {str(e)}")


def __dir__():
    """List available infrastructure modules."""
    return sorted(__all__)
