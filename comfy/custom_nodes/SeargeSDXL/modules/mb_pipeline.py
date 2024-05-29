"""

Custom nodes for SDXL in ComfyUI

MIT License

Copyright (c) 2023 Searge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from .data_utils import retrieve_parameter
from .names import Names
from .ui import UI


# ====================================================================================================
# Magic Box Pipeline
# ====================================================================================================

class Pipeline:
    def __init__(self):
        self.data = None

        self.old_settings = {}
        self.new_settings = {}
        self.old_overrides = {}
        self.new_overrides = {}
        self.cache = {}

        self.pipeline = {}

        self.enabled = True

    def enable(self, enabled=True):
        if self.data is not None:
            self.data[Names.B_MAGIC_BOX_ENABLED] = enabled
            self.enabled = enabled

    def start(self, data):
        self.data = data
        if data is None:
            print("Warning: no data stream for pipeline at start")
            return

        if self.enabled:
            self.old_settings = self.new_settings
            self.old_overrides = self.new_overrides

            for k in self.cache.keys():
                self.cache[k]["changed"] = False

        self.new_settings = {}

        for k in UI.ALL_UI_INPUTS:
            self.new_settings[k] = retrieve_parameter(k, data)

        self.new_overrides = {}

        self.pipeline = {
            "old_settings": self.old_settings,
            "new_settings": self.new_settings,
            "old_overrides": self.old_overrides,
            "new_overrides": self.new_overrides,
            "cache": self.cache,
            "stream": {}
        }

        if data is not None:
            data[PipelineAccess.NAME] = self.pipeline


# ====================================================================================================
# Pipeline Access
# ====================================================================================================

class PipelineAccess:
    NAME = "pipeline"

    def __init__(self, data):
        self.data = data
        self.pipeline = data[PipelineAccess.NAME] if PipelineAccess.NAME in data else None

        if self.pipeline is None:
            print("Warning: pipeline access could not find data")

    def terminate_pipeline(self):
        if self.data is None:
            print("Warning: no data stream for pipeline to terminate")
            return

        if self.pipeline is None:
            print("Warning: no pipeline to terminate in data stream")
            return

        self.pipeline["stream"] = {}

    def is_pipeline_enabled(self):
        return retrieve_parameter(Names.B_MAGIC_BOX_ENABLED, self.data, True)

    # -----===== settings =====-----

    def has_structure(self, name):
        if self.pipeline is None:
            return False

        if name in retrieve_parameter("new_overrides", self.pipeline, {}):
            return True

        if name in retrieve_parameter("new_settings", self.pipeline, {}):
            return True

        return False

    def get_effective_structure(self, name):
        if not self.has_structure(name):
            return None

        new_settings = retrieve_parameter("new_settings", self.pipeline)
        new_overrides = retrieve_parameter("new_overrides", self.pipeline, {})

        new_structure = retrieve_parameter(name, new_settings)
        new_structure_overrides = retrieve_parameter(name, new_overrides, {})
        return new_structure | new_structure_overrides if new_structure is not None else new_structure_overrides

    def override_setting(self, structure_name, field_name, value):
        if self.pipeline is None:
            return False

        new_overrides = retrieve_parameter("new_overrides", self.pipeline, {})
        if structure_name in new_overrides:
            new_overrides[structure_name][field_name] = value
            return True

        new_overrides[structure_name] = {
            field_name: value,
        }

        self.pipeline["new_overrides"] = new_overrides
        return True

    def get_active_setting(self, structure_name, field_name, default=None):
        structure = retrieve_parameter(structure_name, retrieve_parameter("new_overrides", self.pipeline), {})

        if field_name not in structure:
            structure = retrieve_parameter(structure_name, retrieve_parameter("new_settings", self.pipeline))

        field = retrieve_parameter(field_name, structure)
        if default is not None and field is None:
            field = default

        return field

    def get_old_setting(self, structure_name, field_name):
        structure = retrieve_parameter(structure_name, retrieve_parameter("old_overrides", self.pipeline, {}), {})

        if field_name not in structure:
            structure = retrieve_parameter(structure_name, retrieve_parameter("old_settings", self.pipeline))

        return retrieve_parameter(field_name, structure)

    def has_setting(self, structure_name, field_name):
        return self.get_active_setting(structure_name, field_name) is not None

    def setting_changed(self, structure_name, field_name):
        old_value = self.get_old_setting(structure_name, field_name)
        new_value = self.get_active_setting(structure_name, field_name)
        return old_value != new_value

    # -----===== pipeline stream =====-----

    def update_in_pipeline(self, name, value, allow_none=False):
        if self.pipeline is None or "stream" not in self.pipeline:
            return False

        if value is None and not allow_none:
            return False

        self.pipeline["stream"][name] = {
            "changed": True,
            "data": value,
        }

        return True

    def restore_in_pipeline(self, name, value, allow_none=False):
        if self.pipeline is None or "stream" not in self.pipeline:
            return False

        if value is None and not allow_none:
            return False

        self.pipeline["stream"][name] = {
            "changed": False,
            "data": value,
        }

        return True

    def has_in_pipeline(self, name):
        if self.pipeline is not None and "stream" in self.pipeline:
            cache = self.pipeline["stream"]
            if name in cache:
                return True

        return False

    def get_from_pipeline(self, name):
        if self.has_in_pipeline(name):
            cache = self.pipeline["stream"]
            cached = cache[name]
            if "data" in cached:
                return cached["data"]

        return None

    def changed_in_pipeline(self, name):
        if self.has_in_pipeline(name):
            cache = self.pipeline["stream"]
            cached = cache[name]
            return "changed" in cached and cached["changed"]

        return False

    # -----===== cache =====-----

    def update_in_cache(self, name, key, value):
        if self.pipeline is None or "cache" not in self.pipeline:
            return False

        self.pipeline["cache"][name] = {
            "key": key,
            "data": value,
        }

        return True

    def has_in_cache(self, name):
        if self.pipeline is not None and "cache" in self.pipeline:
            cache = self.pipeline["cache"]
            if name in cache:
                return True

        return False

    def get_from_cache(self, name):
        if self.has_in_cache(name):
            cache = self.pipeline["cache"]
            cached = cache[name]
            if "data" in cached:
                return cached["data"]

        return None

    def remove_from_cache(self, name):
        if self.has_in_cache(name):
            cache = self.pipeline["cache"]
            cache.pop(name)
            return True

        return False

    def changed_in_cache(self, name, key):
        if self.has_in_cache(name):
            cache = self.pipeline["cache"]
            cached = cache[name]
            return "key" not in cached or cached["key"] != key

        return True
