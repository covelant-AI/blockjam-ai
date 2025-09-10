from marshmallow import Schema, fields
from .section import SectionSchema

class CourtKeypointsRequest(Schema):
    video_id = fields.Integer(required=True)
    video_url = fields.Url(required=True)
    sections = fields.List(fields.Nested(SectionSchema))