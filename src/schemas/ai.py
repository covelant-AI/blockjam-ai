from pydantic import BaseModel, ConfigDict
from ai_core.player_tracker import PlayerTracker
from ai_core.ball_tracker import BallTracker
from ai_core.court_line_detector import CourtLineDetector
from ai_core.ball_bounce_model import BallBounceModel
from ai_core.mini_court import MiniCourt
from ai_core.scoreboard import Scoreboard
from ai_core.letr_court_line_detector import LETRCourtDetector
from ai_core.racket_player_ball_detector import RacketPlayerBallDetector
from ai_core.shot_type_classifier import ShotTypeClassifier

class AIModels(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    player_tracker: PlayerTracker
    ball_tracker: BallTracker
    racket_player_ball_detector: RacketPlayerBallDetector
    court_line_detector: CourtLineDetector
    letr_court_line_detector: LETRCourtDetector
    ball_bounce_model: BallBounceModel
    mini_court: MiniCourt
    scoreboard_model: Scoreboard
    shot_type_classifier: ShotTypeClassifier