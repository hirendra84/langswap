import cattr
import json

import cattrs
from kombu.serialization import registry

from src.pipeline_models.models import VideoTranslation, RemoteFile


class AttrsSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (VideoTranslation, RemoteFile, )):
            return cattr.unstructure(obj) | {'__type__': type(obj).__name__}
        else:
            return json.JSONEncoder.default(self, obj)


def attrs_decoder(obj):
    models = {
        VideoTranslation.__name__: VideoTranslation,
        RemoteFile.__name__: RemoteFile,
    }
    if '__type__' in obj:
        if obj['__type__'] in models:
            cls = models[obj['__type__']]
            return cattrs.structure(obj, cls)
    return obj


def attrs_dumps(obj):
    return json.dumps(obj, cls=AttrsSerializer)


def attrs_loads(obj):
    return json.loads(obj, object_hook=attrs_decoder)


registry.register(
    'attrs_json',
    attrs_dumps,
    attrs_loads,
    content_type='application/x-attrs-json',
    content_encoding='utf-8'
)
