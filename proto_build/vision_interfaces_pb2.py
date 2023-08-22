# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: vision_interfaces.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import proto_build.sensor_msgs_pb2 as sensor__msgs__pb2
import proto_build.camera3d_interfaces_pb2 as camera3d__interfaces__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='vision_interfaces.proto',
  package='vision_interfaces',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17vision_interfaces.proto\x12\x11vision_interfaces\x1a\x11sensor_msgs.proto\x1a\x19\x63\x61mera3d_interfaces.proto\"\x8d\x01\n\x13\x44\x65tectionErrorCodes\x12?\n\x05value\x18\x01 \x01(\x0e\x32\x30.vision_interfaces.DetectionErrorCodes.ErrorCode\"5\n\tErrorCode\x12\x08\n\x04NONE\x10\x00\x12\x0b\n\x07SUCCESS\x10\x01\x12\x11\n\x04\x46\x41IL\x10\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01\"\xf0\x01\n\x16ItemClassificationInfo\x12<\n\x04type\x18\x01 \x01(\x0e\x32..vision_interfaces.ItemClassificationInfo.Type\x12\r\n\x05score\x18\x02 \x01(\x02\"\x88\x01\n\x04Type\x12\x08\n\x04NONE\x10\x00\x12\x07\n\x03\x42OX\x10\x01\x12\x0c\n\x08\x45NVELOPE\x10\x02\x12\x0b\n\x07PACKAGE\x10\x03\x12\x08\n\x04TEXT\x10\x04\x12\x0b\n\x07\x42\x41RCODE\x10\x05\x12\n\n\x06PALLET\x10\x06\x12\x08\n\x04TOTE\x10\x07\x12\n\n\x06PERSON\x10\x08\x12\x0b\n\x07VEHICLE\x10\t\x12\x0c\n\x08OBSTACLE\x10\n\"\x1d\n\x05Pixel\x12\t\n\x01x\x18\x01 \x01(\r\x12\t\n\x01y\x18\x02 \x01(\r\"4\n\x07\x43ontour\x12)\n\x07\x63ontour\x18\x01 \x03(\x0b\x32\x18.vision_interfaces.Pixel\"\x9d\x02\n\x0eSurfaceSegment\x12\x34\n\x04type\x18\x01 \x01(\x0e\x32&.vision_interfaces.SurfaceSegment.Type\x12\x14\n\x0c\x63oefficients\x18\x02 \x03(\x01\x12\x12\n\ndimensions\x18\x03 \x03(\x01\"\xaa\x01\n\x04Type\x12\t\n\x05PLANE\x10\x00\x12\x07\n\x03\x42OX\x10\x01\x12\n\n\x06\x43IRCLE\x10\x02\x12\x0b\n\x07\x45LLIPSE\x10\x03\x12\x0e\n\nBOX_LENGTH\x10\x00\x12\r\n\tBOX_WIDTH\x10\x01\x12\x0b\n\x07\x42OX_ROT\x10\x02\x12\x0e\n\nCIRCLE_RAD\x10\x00\x12\x11\n\rELLIPSE_MAJOR\x10\x00\x12\x11\n\rELLIPSE_MINOR\x10\x01\x12\x0f\n\x0b\x45LLIPSE_ROT\x10\x02\x1a\x02\x10\x01\"\xc1\x02\n\x0f\x44\x65tectionResult\x12\x41\n\x0e\x63lassification\x18\x01 \x01(\x0b\x32).vision_interfaces.ItemClassificationInfo\x12+\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32\x1d.sensor_msgs.RegionOfInterest\x12,\n\x08\x63ontours\x18\x03 \x03(\x0b\x32\x1a.vision_interfaces.Contour\x12 \n\x04mask\x18\x04 \x01(\x0b\x32\x12.sensor_msgs.Image\x12\x0f\n\x07indices\x18\x05 \x03(\r\x12)\n\x07segment\x18\x06 \x01(\x0b\x32\x18.sensor_msgs.PointCloud2\x12\x32\n\x07surface\x18\x07 \x01(\x0b\x32!.vision_interfaces.SurfaceSegment\"\xb9\x01\n\x12\x44\x65tectionDataTypes\x12\x39\n\x05types\x18\x01 \x03(\x0e\x32*.vision_interfaces.DetectionDataTypes.Type\"h\n\x04Type\x12\x17\n\x13\x43LASSIFICATION_INFO\x10\x00\x12\x08\n\x04\x42\x42OX\x10\x01\x12\x08\n\x04MASK\x10\x02\x12\x0b\n\x07INDICES\x10\x03\x12\x0b\n\x07SURFACE\x10\x04\x12\x0c\n\x08\x43ONTOURS\x10\x64\x12\x0b\n\x07SEGMENT\x10\x65\"\xbc\x01\n\x19\x44\x65tectClassifyItemRequest\x12\x38\n\x0b\x63\x61mera_data\x18\x01 \x01(\x0b\x32#.camera3d_interfaces.RGBDCameraData\x12*\n\x03roi\x18\x02 \x01(\x0b\x32\x1d.sensor_msgs.RegionOfInterest\x12\x39\n\ndata_types\x18\x03 \x01(\x0b\x32%.vision_interfaces.DetectionDataTypes\"\x9c\x01\n\x1a\x44\x65tectClassifyItemResponse\x12:\n\nerror_code\x18\x01 \x01(\x0b\x32&.vision_interfaces.DetectionErrorCodes\x12\x42\n\x16\x64\x65tection_result_array\x18\x02 \x03(\x0b\x32\".vision_interfaces.DetectionResult2{\n\x12\x44\x65tectClassifyItem\x12\x65\n\x04\x43\x61ll\x12,.vision_interfaces.DetectClassifyItemRequest\x1a-.vision_interfaces.DetectClassifyItemResponse\"\x00\x62\x06proto3'
  ,
  dependencies=[sensor__msgs__pb2.DESCRIPTOR,camera3d__interfaces__pb2.DESCRIPTOR,])



_DETECTIONERRORCODES_ERRORCODE = _descriptor.EnumDescriptor(
  name='ErrorCode',
  full_name='vision_interfaces.DetectionErrorCodes.ErrorCode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAIL', index=2, number=-1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=181,
  serialized_end=234,
)
_sym_db.RegisterEnumDescriptor(_DETECTIONERRORCODES_ERRORCODE)

_ITEMCLASSIFICATIONINFO_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='vision_interfaces.ItemClassificationInfo.Type',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NONE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOX', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ENVELOPE', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PACKAGE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TEXT', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BARCODE', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PALLET', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TOTE', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PERSON', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VEHICLE', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OBSTACLE', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=341,
  serialized_end=477,
)
_sym_db.RegisterEnumDescriptor(_ITEMCLASSIFICATIONINFO_TYPE)

_SURFACESEGMENT_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='vision_interfaces.SurfaceSegment.Type',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PLANE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOX', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CIRCLE', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ELLIPSE', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOX_LENGTH', index=4, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOX_WIDTH', index=5, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BOX_ROT', index=6, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CIRCLE_RAD', index=7, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ELLIPSE_MAJOR', index=8, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ELLIPSE_MINOR', index=9, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ELLIPSE_ROT', index=10, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=b'\020\001',
  serialized_start=680,
  serialized_end=850,
)
_sym_db.RegisterEnumDescriptor(_SURFACESEGMENT_TYPE)

_DETECTIONDATATYPES_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='vision_interfaces.DetectionDataTypes.Type',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='CLASSIFICATION_INFO', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BBOX', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='MASK', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INDICES', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SURFACE', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='CONTOURS', index=5, number=100,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SEGMENT', index=6, number=101,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1258,
  serialized_end=1362,
)
_sym_db.RegisterEnumDescriptor(_DETECTIONDATATYPES_TYPE)


_DETECTIONERRORCODES = _descriptor.Descriptor(
  name='DetectionErrorCodes',
  full_name='vision_interfaces.DetectionErrorCodes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='vision_interfaces.DetectionErrorCodes.value', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DETECTIONERRORCODES_ERRORCODE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=234,
)


_ITEMCLASSIFICATIONINFO = _descriptor.Descriptor(
  name='ItemClassificationInfo',
  full_name='vision_interfaces.ItemClassificationInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='vision_interfaces.ItemClassificationInfo.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='score', full_name='vision_interfaces.ItemClassificationInfo.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ITEMCLASSIFICATIONINFO_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=237,
  serialized_end=477,
)


_PIXEL = _descriptor.Descriptor(
  name='Pixel',
  full_name='vision_interfaces.Pixel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='vision_interfaces.Pixel.x', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='vision_interfaces.Pixel.y', index=1,
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
  serialized_start=479,
  serialized_end=508,
)


_CONTOUR = _descriptor.Descriptor(
  name='Contour',
  full_name='vision_interfaces.Contour',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='contour', full_name='vision_interfaces.Contour.contour', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=510,
  serialized_end=562,
)


_SURFACESEGMENT = _descriptor.Descriptor(
  name='SurfaceSegment',
  full_name='vision_interfaces.SurfaceSegment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='vision_interfaces.SurfaceSegment.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='coefficients', full_name='vision_interfaces.SurfaceSegment.coefficients', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dimensions', full_name='vision_interfaces.SurfaceSegment.dimensions', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SURFACESEGMENT_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=565,
  serialized_end=850,
)


_DETECTIONRESULT = _descriptor.Descriptor(
  name='DetectionResult',
  full_name='vision_interfaces.DetectionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='classification', full_name='vision_interfaces.DetectionResult.classification', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bbox', full_name='vision_interfaces.DetectionResult.bbox', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='contours', full_name='vision_interfaces.DetectionResult.contours', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask', full_name='vision_interfaces.DetectionResult.mask', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='indices', full_name='vision_interfaces.DetectionResult.indices', index=4,
      number=5, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='segment', full_name='vision_interfaces.DetectionResult.segment', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='surface', full_name='vision_interfaces.DetectionResult.surface', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=853,
  serialized_end=1174,
)


_DETECTIONDATATYPES = _descriptor.Descriptor(
  name='DetectionDataTypes',
  full_name='vision_interfaces.DetectionDataTypes',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='types', full_name='vision_interfaces.DetectionDataTypes.types', index=0,
      number=1, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _DETECTIONDATATYPES_TYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1177,
  serialized_end=1362,
)


_DETECTCLASSIFYITEMREQUEST = _descriptor.Descriptor(
  name='DetectClassifyItemRequest',
  full_name='vision_interfaces.DetectClassifyItemRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='camera_data', full_name='vision_interfaces.DetectClassifyItemRequest.camera_data', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='roi', full_name='vision_interfaces.DetectClassifyItemRequest.roi', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_types', full_name='vision_interfaces.DetectClassifyItemRequest.data_types', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  serialized_start=1365,
  serialized_end=1553,
)


_DETECTCLASSIFYITEMRESPONSE = _descriptor.Descriptor(
  name='DetectClassifyItemResponse',
  full_name='vision_interfaces.DetectClassifyItemResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_code', full_name='vision_interfaces.DetectClassifyItemResponse.error_code', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='detection_result_array', full_name='vision_interfaces.DetectClassifyItemResponse.detection_result_array', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
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
  serialized_start=1556,
  serialized_end=1712,
)

_DETECTIONERRORCODES.fields_by_name['value'].enum_type = _DETECTIONERRORCODES_ERRORCODE
_DETECTIONERRORCODES_ERRORCODE.containing_type = _DETECTIONERRORCODES
_ITEMCLASSIFICATIONINFO.fields_by_name['type'].enum_type = _ITEMCLASSIFICATIONINFO_TYPE
_ITEMCLASSIFICATIONINFO_TYPE.containing_type = _ITEMCLASSIFICATIONINFO
_CONTOUR.fields_by_name['contour'].message_type = _PIXEL
_SURFACESEGMENT.fields_by_name['type'].enum_type = _SURFACESEGMENT_TYPE
_SURFACESEGMENT_TYPE.containing_type = _SURFACESEGMENT
_DETECTIONRESULT.fields_by_name['classification'].message_type = _ITEMCLASSIFICATIONINFO
_DETECTIONRESULT.fields_by_name['bbox'].message_type = sensor__msgs__pb2._REGIONOFINTEREST
_DETECTIONRESULT.fields_by_name['contours'].message_type = _CONTOUR
_DETECTIONRESULT.fields_by_name['mask'].message_type = sensor__msgs__pb2._IMAGE
_DETECTIONRESULT.fields_by_name['segment'].message_type = sensor__msgs__pb2._POINTCLOUD2
_DETECTIONRESULT.fields_by_name['surface'].message_type = _SURFACESEGMENT
_DETECTIONDATATYPES.fields_by_name['types'].enum_type = _DETECTIONDATATYPES_TYPE
_DETECTIONDATATYPES_TYPE.containing_type = _DETECTIONDATATYPES
_DETECTCLASSIFYITEMREQUEST.fields_by_name['camera_data'].message_type = camera3d__interfaces__pb2._RGBDCAMERADATA
_DETECTCLASSIFYITEMREQUEST.fields_by_name['roi'].message_type = sensor__msgs__pb2._REGIONOFINTEREST
_DETECTCLASSIFYITEMREQUEST.fields_by_name['data_types'].message_type = _DETECTIONDATATYPES
_DETECTCLASSIFYITEMRESPONSE.fields_by_name['error_code'].message_type = _DETECTIONERRORCODES
_DETECTCLASSIFYITEMRESPONSE.fields_by_name['detection_result_array'].message_type = _DETECTIONRESULT
DESCRIPTOR.message_types_by_name['DetectionErrorCodes'] = _DETECTIONERRORCODES
DESCRIPTOR.message_types_by_name['ItemClassificationInfo'] = _ITEMCLASSIFICATIONINFO
DESCRIPTOR.message_types_by_name['Pixel'] = _PIXEL
DESCRIPTOR.message_types_by_name['Contour'] = _CONTOUR
DESCRIPTOR.message_types_by_name['SurfaceSegment'] = _SURFACESEGMENT
DESCRIPTOR.message_types_by_name['DetectionResult'] = _DETECTIONRESULT
DESCRIPTOR.message_types_by_name['DetectionDataTypes'] = _DETECTIONDATATYPES
DESCRIPTOR.message_types_by_name['DetectClassifyItemRequest'] = _DETECTCLASSIFYITEMREQUEST
DESCRIPTOR.message_types_by_name['DetectClassifyItemResponse'] = _DETECTCLASSIFYITEMRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DetectionErrorCodes = _reflection.GeneratedProtocolMessageType('DetectionErrorCodes', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONERRORCODES,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.DetectionErrorCodes)
  })
_sym_db.RegisterMessage(DetectionErrorCodes)

ItemClassificationInfo = _reflection.GeneratedProtocolMessageType('ItemClassificationInfo', (_message.Message,), {
  'DESCRIPTOR' : _ITEMCLASSIFICATIONINFO,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.ItemClassificationInfo)
  })
_sym_db.RegisterMessage(ItemClassificationInfo)

Pixel = _reflection.GeneratedProtocolMessageType('Pixel', (_message.Message,), {
  'DESCRIPTOR' : _PIXEL,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.Pixel)
  })
_sym_db.RegisterMessage(Pixel)

Contour = _reflection.GeneratedProtocolMessageType('Contour', (_message.Message,), {
  'DESCRIPTOR' : _CONTOUR,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.Contour)
  })
_sym_db.RegisterMessage(Contour)

SurfaceSegment = _reflection.GeneratedProtocolMessageType('SurfaceSegment', (_message.Message,), {
  'DESCRIPTOR' : _SURFACESEGMENT,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.SurfaceSegment)
  })
_sym_db.RegisterMessage(SurfaceSegment)

DetectionResult = _reflection.GeneratedProtocolMessageType('DetectionResult', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONRESULT,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.DetectionResult)
  })
_sym_db.RegisterMessage(DetectionResult)

DetectionDataTypes = _reflection.GeneratedProtocolMessageType('DetectionDataTypes', (_message.Message,), {
  'DESCRIPTOR' : _DETECTIONDATATYPES,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.DetectionDataTypes)
  })
_sym_db.RegisterMessage(DetectionDataTypes)

DetectClassifyItemRequest = _reflection.GeneratedProtocolMessageType('DetectClassifyItemRequest', (_message.Message,), {
  'DESCRIPTOR' : _DETECTCLASSIFYITEMREQUEST,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.DetectClassifyItemRequest)
  })
_sym_db.RegisterMessage(DetectClassifyItemRequest)

DetectClassifyItemResponse = _reflection.GeneratedProtocolMessageType('DetectClassifyItemResponse', (_message.Message,), {
  'DESCRIPTOR' : _DETECTCLASSIFYITEMRESPONSE,
  '__module__' : 'vision_interfaces_pb2'
  # @@protoc_insertion_point(class_scope:vision_interfaces.DetectClassifyItemResponse)
  })
_sym_db.RegisterMessage(DetectClassifyItemResponse)


_SURFACESEGMENT_TYPE._options = None

_DETECTCLASSIFYITEM = _descriptor.ServiceDescriptor(
  name='DetectClassifyItem',
  full_name='vision_interfaces.DetectClassifyItem',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1714,
  serialized_end=1837,
  methods=[
  _descriptor.MethodDescriptor(
    name='Call',
    full_name='vision_interfaces.DetectClassifyItem.Call',
    index=0,
    containing_service=None,
    input_type=_DETECTCLASSIFYITEMREQUEST,
    output_type=_DETECTCLASSIFYITEMRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_DETECTCLASSIFYITEM)

DESCRIPTOR.services_by_name['DetectClassifyItem'] = _DETECTCLASSIFYITEM

# @@protoc_insertion_point(module_scope)
