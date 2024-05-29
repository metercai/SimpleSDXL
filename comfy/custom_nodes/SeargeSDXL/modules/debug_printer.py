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

from .names import Names
from .ui import UI


# ====================================================================================================
# Print state of a data stream
# ====================================================================================================

class SeargeDebugPrinter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True},),
            },
            "optional": {
                "data": ("SRG_DATA_STREAM",),
                "prefix": ("STRING", {"multiline": False, "default": ""},),
            },
        }

    RETURN_TYPES = ("SRG_DATA_STREAM",)
    RETURN_NAMES = ("data",)
    FUNCTION = "output"

    OUTPUT_NODE = True

    CATEGORY = UI.CATEGORY_DEBUG

    def output(self, enabled, data=None, prefix=None):
        if data is None or not enabled:
            return (data,)

        prefix = "" if prefix is None or len(prefix) < 1 else prefix + ": "

        indent_spaces = "· "

        test_data = False
        if test_data:
            data["test_dict"] = {"k1": 1.0, "k2": 2, "k3": True}
            data["test_list"] = ["l1", 2.0, 3]
            data["test_tuple"] = (1, "t2", 3.0)

        def print_dict(coll, ind=0, kp='"', pk=True):
            spaces = indent_spaces * ind
            for (k, v) in coll.items():
                print_val(k, v, ind, kp, pk)

        def print_coll(coll, ind=0, kp='', pk=False):
            spaces = indent_spaces * ind
            cl = len(coll)
            for i in range(0, cl):
                v = coll[i]
                print_val(i, v, ind, kp, pk)

        def print_val(k, v, ind=0, kp='"', pk=True):
            spaces = indent_spaces * ind
            key = kp + str(k) + kp + ': ' if pk else ''

            if ind > 10:
                print(prefix + spaces + key + '<max recursion depth>')
                return

            if v is None:
                print(prefix + spaces + key + 'None,')
            elif isinstance(v, int) or isinstance(v, float):
                print(prefix + spaces + key + str(v) + ',')
            elif isinstance(v, str):
                print(prefix + spaces + key + '"' + v + '",')
            elif isinstance(v, dict):
                # dirty hack: we don't need to print the whole workflow and prompt
                if k != Names.S_MAGIC_BOX_HIDDEN:
                    print(prefix + spaces + key + '{')
                    print_dict(v, ind + 1, '"', True)
                    print(prefix + spaces + '},')
                else:
                    print(prefix + spaces + key + '{ ... printing skipped ... }')
            elif isinstance(v, list):
                print(prefix + spaces + key + '[')
                print_coll(v, ind + 1, '', True)
                print(prefix + spaces + '],')
            elif isinstance(v, tuple):
                print(prefix + spaces + key + '(')
                print_coll(v, ind + 1, '', False)
                print(prefix + spaces + '),')
            else:
                print(prefix + spaces + key + str(type(v)))

        print(prefix + "===============================================================================")
        if not isinstance(data, dict):
            print(prefix + " ! invalid data stream !")
        else:
            print(prefix + "* DATA STREAM *")
            print(prefix + "---------------")
            print_val("data", data)
        print(prefix + "===============================================================================")

        return (data,)
