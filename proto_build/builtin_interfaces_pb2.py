# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: builtin_interfaces.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='builtin_interfaces.proto',
  package='builtin_interfaces',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x18\x62uiltin_interfaces.proto\x12\x12\x62uiltin_interfaces\"$\n\x04Time\x12\x0b\n\x03sec\x18\x01 \x01(\x05\x12\x0f\n\x07nanosec\x18\x02 \x01(\rb\x06proto3'
)




_TIME = _descriptor.Descriptor(
  name='Time',
  full_name='builtin_interfaces.Time',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='sec', full_name='builtin_interfaces.Time.sec', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='nanosec', full_name='builtin_interfaces.Time.nanosec', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=48,
  serialized_end=84,
)

DESCRIPTOR.message_types_by_name['Time'] = _TIME
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Time = _reflection.GeneratedProtocolMessageType('Time', (_message.Message,), {
  'DESCRIPTOR' : _TIME,
  '__module__' : 'builtin_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:builtin_interfaces.Time)
  })
_sym_db.RegisterMessage(Time)


# @@protoc_insertion_point(module_scope)
