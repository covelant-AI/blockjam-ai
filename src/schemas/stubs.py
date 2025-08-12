from marshmallow import Schema, fields
from schemas.section import SectionSchema, ErrorSection

class StubsRequest(Schema):
    video_id = fields.Integer(required=True, error_messages={'required': 'Video ID is required'})
    video_url = fields.Url(required=True, error_messages={'required': 'Video URL is required'})
    sections = fields.List(fields.Nested(SectionSchema), required=True, error_messages={'required': 'Sections are required'})

class PlayerStubsRequest(StubsRequest):
    all_court_keypoints = fields.List(fields.Raw(), required=True, error_messages={'required': 'All court keypoints are required'})

class StubsData(Schema):
    section = fields.Nested(SectionSchema)
    data = fields.Raw()

class StubsResponse(Schema):
    video_id = fields.String()
    data = fields.List(fields.Nested(StubsData))
    error_sections = fields.List(fields.Nested(ErrorSection))


class VideoInfo(Schema):
    fps = fields.Float()
    width = fields.Integer()
    height = fields.Integer()
    total_frames = fields.Integer()

class SectionedStubsData(Schema):
    section = fields.Nested(SectionSchema)
    player_detections = fields.List(fields.Raw(), required=False)
    ball_detections = fields.List(fields.Raw(), required=False)
    court_keypoints = fields.List(fields.Raw(), required=True)