from marshmallow import Schema, fields
from schemas.section import SectionSchema, ErrorSection
from schemas.stubs import VideoInfo, SectionedStubsData

class PlayerSpeedsRequest(Schema):
    video_id = fields.Integer(required=True, error_messages={'required': 'Video ID is required'})
    video_info = fields.Nested(VideoInfo, required=True, error_messages={'required': 'Video info is required'})
    sectioned_data = fields.List(fields.Nested(SectionedStubsData), required=True, error_messages={'required': 'Sectioned data is required'})

class PlayerSpeedsForSections(Schema):
    section = fields.Nested(SectionSchema)
    p1_speeds = fields.List(fields.Float())
    p2_speeds = fields.List(fields.Float())

class PlayerSpeedResponse(Schema):
    video_id = fields.String()
    time_step = fields.Float()
    data = fields.List(fields.Nested(PlayerSpeedsForSections))
    error_sections = fields.List(fields.Nested(ErrorSection))