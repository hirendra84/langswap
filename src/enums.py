from enum import Enum


class ProcessStatus(str, Enum):
    uploaded = 'uploaded'
    in_progress = 'in_progress'
    translation_ready = 'translation_ready'
    done = 'done'
    error = 'error'
    removed = 'removed'
