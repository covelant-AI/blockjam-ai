from pydantic import BaseModel, ConfigDict
from ai_core.trackers.core import BallAndPlayerTracker
from ai_core.court_line_detector import CourtLineDetector
from ai_core.letr_court_line_detector import LETRCourtDetector

class AIModels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ball_and_player_tracker: BallAndPlayerTracker
    court_line_detector: CourtLineDetector
    letr_court_line_detector: LETRCourtDetector