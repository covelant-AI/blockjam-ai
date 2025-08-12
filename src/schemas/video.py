from marshmallow import Schema, fields, ValidationError
from enum import Enum
from typing import List, Set


class Feature(Enum):
    DEAD_TIME_DETECTION = "DeadTimeDetection"
    MATCH_SECTIONING = "MatchSectioning"
    TRACK_BALL_SPEED = "TrackBallSpeed"
    TRACK_PLAYER_SPEED = "TrackPlayerSpeed"
    CLASSIFY_SHOT_TYPE = "ClassifyShotType"

# Define parent feature dependencies
FEATURE_DEPENDENCIES = {
    Feature.MATCH_SECTIONING: {Feature.DEAD_TIME_DETECTION},
    Feature.TRACK_BALL_SPEED: {Feature.MATCH_SECTIONING, Feature.DEAD_TIME_DETECTION},
    Feature.TRACK_PLAYER_SPEED: {Feature.MATCH_SECTIONING, Feature.DEAD_TIME_DETECTION},
    Feature.CLASSIFY_SHOT_TYPE: {Feature.MATCH_SECTIONING, Feature.DEAD_TIME_DETECTION},
}


def validate_features(features: List[str]) -> List[str]:
    """Validate features and ensure parent dependencies are met."""
    valid_features = [feature.value for feature in Feature]
    
    # Check if all features are valid
    for feature in features:
        if feature not in valid_features:
            raise ValidationError(f"Invalid feature: {feature}")
    
    # Check parent dependencies
    feature_set = set(features)
    for feature_name in features:
        feature_enum = Feature(feature_name)
        if feature_enum in FEATURE_DEPENDENCIES:
            required_parents = FEATURE_DEPENDENCIES[feature_enum]
            # Convert feature_set to Feature enum objects for comparison
            feature_enums = {Feature(f) for f in features}
            missing_parents = required_parents - feature_enums
            if missing_parents:
                missing_names = [parent.value for parent in missing_parents]
                raise ValidationError(
                    f"Feature '{feature_name}' requires parent features: {', '.join(missing_names)}"
                )
    
    return features


# Define your schema
class VideoRequestSchema(Schema):
    video_url = fields.Url(required=True, error_messages={'required': 'Video URL is required'})
    video_id = fields.Integer(required=True, error_messages={'required': 'Video ID is required'})
    features = fields.List(fields.String(), required=True, error_messages={'required': 'Features are required'}, validate=validate_features)