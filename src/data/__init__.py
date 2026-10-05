"""Dataset loaders. All CPU/network work: downloading, parsing annotations,
extracting clips. No GPU needed for anything in this module."""

from .sportsmot import SportsMOT
from .soccernet import SoccerNetTracking

__all__ = ["SportsMOT", "SoccerNetTracking"]
