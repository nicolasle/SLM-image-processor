"""Widgets package for SLM Image Processor."""

from slm_processor.widgets.image_canvas import SLMImageCanvas
from slm_processor.widgets.profile_canvas import SLMProfileCanvas
from slm_processor.widgets.series_canvas import SLMSeriesCanvas
from slm_processor.widgets.dual_view import SLMDualView
from slm_processor.widgets.overlay_view import SLMOverlayView
from slm_processor.widgets.channel_manager_dialog import SLMChannelManagerDialog
from slm_processor.widgets.hermitian_dialog import SLMHermitianDialog
from slm_processor.widgets.erf_fit_dialog import SLMErfFitDialog

__all__ = [
    "SLMImageCanvas",
    "SLMProfileCanvas",
    "SLMSeriesCanvas",
    "SLMDualView",
    "SLMOverlayView",
    "SLMChannelManagerDialog",
    "SLMHermitianDialog",
    "SLMErfFitDialog",
]
