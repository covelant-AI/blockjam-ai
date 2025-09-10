from marshmallow import Schema, fields
from schemas.section import SectionSchema, ErrorSection
from schemas.stubs import VideoInfo, SectionedStubsData

class BallSpeedsRequest(Schema):
    video_id = fields.Integer(required=True, error_messages={'required': 'Video ID is required'})
    video_info = fields.Nested(VideoInfo, required=True, error_messages={'required': 'Video info is required'})
    sectioned_data = fields.List(fields.Nested(SectionedStubsData), required=True, error_messages={'required': 'Sectioned data is required'})


class BallSpeedsForSection(Schema):
    section = fields.Nested(SectionSchema)
    speeds = fields.List(fields.Float())

class BallSpeedResponse(Schema):
    video_id = fields.String()
    time_step = fields.Float()
    data = fields.List(fields.Nested(BallSpeedsForSection))
    error_sections = fields.List(fields.Nested(ErrorSection))