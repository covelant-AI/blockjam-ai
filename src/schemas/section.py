from marshmallow import Schema, fields

class SectionPartSchema(Schema):
    index = fields.Int()
    time = fields.Float()

class SectionSchema(Schema):
    start = fields.Nested(SectionPartSchema)
    end = fields.Nested(SectionPartSchema)

class ErrorSection(Schema):
    section = fields.Nested(SectionSchema)
    message = fields.String()