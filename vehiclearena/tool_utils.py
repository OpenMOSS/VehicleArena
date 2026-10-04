"""
Function-calling utilities for VehicleWorld evaluation.

Converts the @api-registered methods into OpenAI-compatible tool schemas
and dispatches tool_call requests to the correct VehicleWorld module methods.

Tool naming: {module}__{method_name}  (double underscore separator)
  e.g. "fogLight__carcontrol_fogLight_switch"

Supports 3 docstring formats for parameter extraction:
  - Args:\n  param (type): desc        (198 methods)
  - Parameters:\n  - param (type): desc (20 methods)
  - :param name: desc                   (4 methods)
"""

import copy
import difflib
import hashlib
import inspect
import json
import re
import threading
import types
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal, get_args, get_origin, get_type_hints
from jsonschema import Draft202012Validator

import utils

# ── External modules: derived from registry (single source of truth) ──
def _get_external_modules():
    from registry import ModuleRegistry
    return set(ModuleRegistry.instance().external_module_names())

# Lazy-load so import order doesn't matter
class _ExternalModulesProxy(set):
    _loaded = False
    def _ensure(self):
        if not self._loaded:
            self.update(_get_external_modules())
            self._loaded = True
    def __contains__(self, item):
        self._ensure(); return super().__contains__(item)
    def __iter__(self):
        self._ensure(); return super().__iter__()
    def __len__(self):
        self._ensure(); return super().__len__()

EXTERNAL_MODULES = _ExternalModulesProxy()

# ── OpenAI function name limit ──
_MAX_TOOL_NAME_LEN = 64


def _make_tool_name(module_name: str, method_name: str) -> str:
    """Build tool name, shortening if it exceeds OpenAI's 64-char limit.

    Method names often follow the pattern ``carcontrol_{module}_{action}``,
    which duplicates the module name already present in the
    ``{module}__{method}`` format.  When the full name is too long, strip
    the ``carcontrol_{module}_`` prefix.
    """
    full = f"{module_name}__{method_name}"
    if len(full) <= _MAX_TOOL_NAME_LEN:
        return full
    prefix = f"carcontrol_{module_name}_"
    if method_name.startswith(prefix):
        short = f"{module_name}__{method_name[len(prefix):]}"
        if len(short) <= _MAX_TOOL_NAME_LEN:
            return short
    # Last resort: truncate (shouldn't happen with current modules)
    return full[:_MAX_TOOL_NAME_LEN]


# A compact public spelling for legacy cabin APIs whose Python method repeats
# the module/device name.  The long spelling remains accepted for backward
# compatibility; discovery advertises the shorter alias to reduce guessing.
_CANONICAL_METHOD_PREFIXES = {
    "HUD": "carcontrol_HUD_",
    "bluetooth": "carcontrol_connection_bluetooth_",
    "centerInformationDisplay": "carcontrol_centerInformationDisplay_",
    "door": "carcontrol_carDoor_",
    "fogLight": "carcontrol_fogLight_",
    "footPedal": "carcontrol_pedals_",
    "frontTrunk": "carcontrol_frontTrunk_",
    "instrumentPanel": "carcontrol_instrumentPanel_",
    "overheadScreen": "carcontrol_overheadScreen_",
    "positionLight": "carcontrol_positionLight_",
    "readingLight": "carcontrol_readingLight_",
    "seat": "carcontrol_carSeat_",
    "steeringWheel": "carcontrol_steeringWheel_",
    "sunroof": "carcontrol_sunroof_",
    "sunshade": "carcontrol_sunshade_",
    "trunk": "carcontrol_trunk_",
    "window": "carcontrol_window_",
    "wiper": "carcontrol_wiperBlade_",
}


def _canonical_tool_name(module_name: str, method_name: str) -> str:
    prefix = _CANONICAL_METHOD_PREFIXES.get(module_name)
    if prefix and method_name.startswith(prefix):
        return _make_tool_name(module_name, method_name[len(prefix):])
    return _make_tool_name(module_name, method_name)


def _expand_canonical_method(module_name: str, method_name: str) -> str:
    prefix = _CANONICAL_METHOD_PREFIXES.get(module_name)
    if prefix and not method_name.startswith("carcontrol_"):
        return prefix + method_name
    return method_name


def _resolve_method_name(module_name: str, method_name: str, module_obj) -> str:
    """Resolve a possibly-shortened method name back to the real attribute.

    Tries ``method_name`` first, then ``carcontrol_{module}_{method_name}``.
    Returns the actual attribute name, or the original if neither matches.
    """
    if hasattr(module_obj, method_name):
        return method_name
    expanded = f"carcontrol_{module_name}_{method_name}"
    if hasattr(module_obj, expanded):
        return expanded
    return method_name  # will raise AttributeError at callsite

# ── Python type string → JSON Schema type ──
_TYPE_MAP = {
    "bool": "boolean",
    "boolean": "boolean",
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "dict": "object",
    "Dict": "object",
    "list": "array",
    "List": "array",
}


def _python_type_to_json_schema(type_str: str) -> Optional[str]:
    """Map a Python type string to a JSON Schema type string.

    Returns None if the type is unrecognized or complex (e.g. Optional[List[str]]).
    """
    if not type_str:
        return None
    # Strip Optional[], optional prefix
    cleaned = type_str.strip()
    cleaned = re.sub(r'^Optional\[(.+)\]$', r'\1', cleaned)
    cleaned = cleaned.split(",")[0].strip()  # take first part if "int, optional"
    cleaned = cleaned.replace("optional", "").strip()
    # Direct lookup
    return _TYPE_MAP.get(cleaned)


def _annotation_to_json_schema(annotation) -> Optional[str]:
    """Convert a Python type annotation object to JSON Schema type string."""
    return _annotation_schema(annotation).get("type")


def _annotation_schema(annotation) -> Dict:
    """Preserve containers and unions; Optional means an omittable value here.

    Requiredness comes from the signature default, not the annotation. Resolve
    postponed annotations against the real method before calling this helper.
    """
    if annotation is None or annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        variants = [_annotation_schema(arg) for arg in args if arg is not type(None)]
        if len(variants) == 1:
            return variants[0]
        return {"anyOf": variants} if variants else {}
    if origin is Literal:
        return {**_annotation_schema(type(args[0])), "enum": list(args)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        values = [item.value for item in annotation]
        return {**_annotation_schema(type(values[0])), "enum": values} if values else {}
    if origin is list or annotation is list:
        return {"type": "array", "items": _annotation_schema(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        return {"type": "object", **({"additionalProperties": _annotation_schema(args[1])} if len(args) == 2 else {})}
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    mapped = _TYPE_MAP.get(name)
    return {"type": mapped} if mapped else {}


def _extract_enum_values(text: str) -> Optional[List[str]]:
    """Extract enum/valid values from a description string.

    Looks for patterns like:
      - enum values: "front", "rear", "all"
      - Valid values are "Kilometers" or "Miles"
      - Possible values: front, rear, all
      - must be selected from the following enumeration values: [open, close]
      - must be selected from the following list: [a, b, c]
    """
    if not text:
        return None

    # Pattern 1: [value1, value2, ...] (bracket list)
    bracket_match = re.search(r'\[([^\]]+)\]', text)
    if bracket_match and not re.search(r'one of|enum|values|list', text, re.I):
        bracket_match = None
    if bracket_match:
        items = [v.strip().strip('"').strip("'") for v in bracket_match.group(1).split(',')]
        items = [v for v in items if v]
        # Deduplicate while preserving order
        seen = set()
        items = [v for v in items if v not in seen and not seen.add(v)]
        if items and len(items) <= 20:
            return items

    # Pattern 2: "val1", "val2", ... or "val1" or "val2"
    quoted = re.findall(r'"([^"]+)"', text)
    if len(quoted) >= 2:
        seen = set()
        quoted = [v for v in quoted if v not in seen and not seen.add(v)]
        return quoted

    # Pattern 3: Possible values: val1, val2, val3
    pv_match = re.search(r'(?:Possible|Valid|Enum)\s+values[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    if pv_match:
        vals_str = pv_match.group(1)
        vals = [v.strip().strip('"').strip("'") for v in re.split(r'[,\s]+(?:or\s+)?', vals_str)]
        vals = [v for v in vals if v and v not in ('are', 'is')]
        seen = set()
        vals = [v for v in vals if v not in seen and not seen.add(v)]
        if vals:
            return vals

    return None


def _parse_docstring_params(docstring: str) -> Dict[str, Dict]:
    """Parse parameter info from docstring.

    Returns: {param_name: {"description": str, "type_str": str|None, "enum": list|None}}
    """
    params = {}
    if not docstring:
        return params

    # Detect which style this docstring uses
    has_args = bool(re.search(r'\bArgs\s*:', docstring))
    has_parameters = bool(re.search(r'\bParameters\s*:', docstring))
    has_sphinx = bool(re.search(r':param\s+', docstring))

    if has_sphinx:
        # Style: :param name: description\n  Possible values: ...
        # Match :param blocks (possibly multi-line until next :param or :return or end)
        pattern = r':param\s+(\w+)\s*:\s*(.*?)(?=\n\s*:|\n\s*$|\Z)'
        for match in re.finditer(pattern, docstring, re.DOTALL):
            name = match.group(1)
            desc_block = match.group(2).strip()
            # Check for Possible values on subsequent lines
            enum_vals = _extract_enum_values(desc_block)
            # Clean description (first line only)
            desc_first_line = desc_block.split('\n')[0].strip()
            params[name] = {
                "description": desc_first_line,
                "type_str": None,
                "enum": enum_vals,
            }

    elif has_parameters:
        # Parameters entries may be bulleted or plain, with either
        # "name: type description" or "name (type): description".
        # Split the Parameters section
        params_section = re.split(r'\bParameters\s*:', docstring, maxsplit=1)
        if len(params_section) > 1:
            section = params_section[1]
            # Truncate at Returns:
            section = re.split(r'\b(?:Returns|Raises|Examples|Note|Notes)\s*:', section, maxsplit=1)[0]
            # Anchor entries to their indentation so continuation lines such
            # as "Enum values:" do not become separate parameters.
            pattern = (
                r'^([ \t]*)(?:-[ \t]*)?(\w+)[ \t]*'
                r'(?::[ \t]*(\w+)[ \t]+(.*?)|\(([^)]+)\)[ \t]*:[ \t]*(.*?))'
                r'(?=\n\1(?:-[ \t]*)?\w+[ \t]*(?:\([^)]*\))?[ \t]*:|\Z)'
            )
            for match in re.finditer(pattern, section, re.DOTALL | re.MULTILINE):
                name = match.group(2)
                if match.group(3):  # "name: type desc" format
                    type_str = match.group(3)
                    desc = match.group(4).strip()
                else:  # "name (type): desc" format
                    type_str = match.group(5).strip()
                    desc = match.group(6).strip()
                enum_vals = _extract_enum_values(desc)
                desc_first_line = desc.split('\n')[0].strip()
                params[name] = {
                    "description": desc_first_line,
                    "type_str": type_str,
                    "enum": enum_vals,
                }

    elif has_args:
        # Style: Args:\n  param (type): desc  OR  param (type, optional): desc
        args_section = re.split(r'\bArgs\s*:', docstring, maxsplit=1)
        if len(args_section) > 1:
            section = args_section[1]
            section = re.split(r'\b(?:Returns|Raises|Examples|Note|Notes)\s*:', section, maxsplit=1)[0]
            # Match: param_name (type_info): description (possibly multi-line)
            pattern = r'^([ \t]+)(\w+)[ \t]*(?:\(([^)]*)\))?[ \t]*:[ \t]*(.*?)(?=\n\1\w+[ \t]*(?:\([^)]*\))?[ \t]*:|\Z)'
            for match in re.finditer(pattern, section, re.DOTALL | re.MULTILINE):
                name = match.group(2)
                type_str = match.group(3).strip() if match.group(3) else None
                desc = match.group(4).strip()
                enum_vals = _extract_enum_values(desc)
                desc_first_line = desc.split('\n')[0].strip()
                params[name] = {
                    "description": desc_first_line,
                    "type_str": type_str,
                    "enum": enum_vals,
                }

    return params


def _build_param_schema(
    param_name: str,
    sig_param: Optional[inspect.Parameter],
    doc_info: Optional[Dict],
) -> Dict:
    """Build JSON Schema for a single parameter, combining signature + docstring info."""
    schema = _annotation_schema(sig_param.annotation) if sig_param else {}

    # 1. Try type from signature annotation
    json_type = None
    if sig_param and sig_param.annotation is not inspect.Parameter.empty:
        json_type = _annotation_to_json_schema(sig_param.annotation)

    # 2. Fallback: type from docstring
    if not json_type and "anyOf" not in schema and doc_info and doc_info.get("type_str"):
        json_type = _python_type_to_json_schema(doc_info["type_str"])

    if json_type:
        schema["type"] = json_type

    # 3. Description from docstring
    if doc_info and doc_info.get("description"):
        schema["description"] = doc_info["description"]

    # 4. Enum values from docstring
    if doc_info and doc_info.get("enum"):
        if schema.get("type") == "array":
            # enum belongs on items, not on the array itself
            pass  # handled in step 5 below
        elif (schema.get("type", "string") == "string"
              and "anyOf" not in schema and "enum" not in schema):
            schema["enum"] = doc_info["enum"]

    # 5. OpenAI requires "items" for array types
    if schema.get("type") == "array":
        items_schema = schema.get("items", {"type": "string"})
        if doc_info and doc_info.get("enum") and items_schema.get("type") == "string":
            items_schema["enum"] = doc_info["enum"]
        schema["items"] = items_schema
        # Remove top-level enum if accidentally set
        schema.pop("enum", None)

    # 6. Fallback: ensure every property has a type
    if not schema or ("type" not in schema and "anyOf" not in schema):
        schema["type"] = "string"

    return schema


def _get_method_object(module_name: str, method_name: str):
    """Try to get the actual method object from a VehicleWorld instance.

    Returns None if not possible (avoids creating VehicleWorld just for inspection).
    We use a lazily-created singleton for inspection only (thread-safe).
    """
    global _inspection_vw
    try:
        if _inspection_vw is None:
            with _inspection_lock:
                if _inspection_vw is None:
                    from vehiclearena import VehicleWorld
                    _inspection_vw = VehicleWorld()
        vw = _inspection_vw
        if module_name in EXTERNAL_MODULES:
            mod = getattr(vw.externalWorld, module_name)
        else:
            mod = getattr(vw, module_name)
        return getattr(mod, method_name)
    except Exception:
        return None

# Module-level inspection instance (lazy, thread-safe)
_inspection_vw = None
_inspection_lock = threading.Lock()


def generate_tools_schema(apis_dict: Optional[Dict] = None, modules: Optional[List[str]] = None) -> List[Dict]:
    """Convert @api-registered methods to OpenAI function-calling tool schemas.

    Args:
        apis_dict: API registry dict. Defaults to the global `apis` from utils.py.
        modules: If provided, only generate schemas for these module names.

    Returns:
        List of tool dicts in OpenAI format:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]
    """
    if apis_dict is None:
        apis_dict = utils.apis  # read at call-time, after modules have been imported

    tools = []

    for module_name, methods in apis_dict.items():
        if modules and module_name not in modules:
            continue

        for method_info in methods:
            method_name = method_info['name']
            source = method_info['source']
            tool_name = _make_tool_name(module_name, method_name)

            # Extract docstring from source
            docstring_match = re.search(r'"""(.*?)"""', source, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else ""

            # First line of docstring as description
            description = docstring.split('\n')[0].strip() if docstring else method_name

            # Parse docstring for parameter details
            doc_params = _parse_docstring_params(docstring)

            # Try to get actual method for inspect.signature
            method_obj = _get_method_object(module_name, method_name)
            sig = None
            if method_obj:
                # The registry's abbreviated source flattens indentation.
                # Prefer the real docstring so continuation lines and section
                # boundaries have exactly the same meaning as at dispatch.
                docstring = inspect.getdoc(method_obj) or docstring
                description = docstring.split('\n')[0].strip() if docstring else method_name
                doc_params = _parse_docstring_params(docstring)
                try:
                    sig = inspect.signature(method_obj)
                    try:
                        hints = get_type_hints(inspect.unwrap(method_obj))
                    except (NameError, TypeError):
                        hints = {}
                    sig = sig.replace(parameters=[
                        param.replace(annotation=hints.get(name, param.annotation))
                        for name, param in sig.parameters.items()])
                except (ValueError, TypeError):
                    pass

            # Build parameters schema
            properties = {}
            required = []

            if sig:
                for pname, param in sig.parameters.items():
                    if pname == 'self':
                        continue
                    doc_info = doc_params.get(pname)
                    param_schema = _build_param_schema(pname, param, doc_info)
                    properties[pname] = param_schema

                    # Required if no default value
                    if param.default is inspect.Parameter.empty:
                        required.append(pname)
            else:
                # Fallback: parse def line from source for parameter names
                def_match = re.match(r'def\s+\w+\s*\(self(?:,\s*(.+))?\)', source.split('\n')[0])
                if def_match and def_match.group(1):
                    raw_params = def_match.group(1)
                    for part in raw_params.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        # Handle "param: type = default" or "param=default"
                        has_default = '=' in part
                        pname = re.split(r'[:\s=]', part)[0].strip()
                        if not pname:
                            continue
                        doc_info = doc_params.get(pname)
                        param_schema = _build_param_schema(pname, None, doc_info)
                        properties[pname] = param_schema
                        if not has_default:
                            required.append(pname)

            parameters_schema = {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            }
            if required:
                parameters_schema["required"] = required

            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters_schema,
                }
            })

    return tools


def generate_discovery_tools() -> List[Dict]:
    """Generate tool schemas for the two discovery APIs: get_modules and get_module_API.

    These are registered first so the agent can discover modules before calling specific APIs.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_modules",
                "description": "Returns a dict of all available module names and their descriptions.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_module_API",
                "description": "Returns detailed API signatures and documentation for the specified modules. Use this to learn what APIs are available before calling them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "modules": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of module names to look up, e.g. ['fogLight', 'wiper']"
                        }
                    },
                    "required": ["modules"]
                }
            }
        }
    ]


def dispatch(vw, tool_name: str, arguments: dict) -> Any:
    """Route a tool_call to the correct VehicleWorld module method and execute it.

    Args:
        vw: VehicleWorld instance
        tool_name: "{module}__{method_name}", e.g. "fogLight__carcontrol_fogLight_switch"
        arguments: dict of keyword arguments, e.g. {"switch": true, "position": "front"}

    Returns:
        The return value of the called method.
    """
    parts = tool_name.split("__", 1)
    if len(parts) != 2:
        return {
            "success": False,
            "error": (
                f"Invalid tool name format: {tool_name}. "
                "Expected 'module__method'."),
        }

    module_name, method_name = parts
    method_name = _expand_canonical_method(module_name, method_name)
    if hasattr(vw, "has_module") and not vw.has_module(module_name):
        return {
            "success": False,
            "error": "capability_not_available",
            "module": module_name,
            "equipment_profile": getattr(
                getattr(vw, "capabilities", None),
                "equipment_profile", ""),
        }

    # Route: external modules via vw.externalWorld, internal modules directly on vw
    try:
        if module_name in EXTERNAL_MODULES:
            module = getattr(vw.externalWorld, module_name)
        else:
            module = getattr(vw, module_name)
    except AttributeError:
        return {
            "success": False,
            "error": f"Module '{module_name}' not found on VehicleWorld.",
        }

    real_method_name = _resolve_method_name(module_name, method_name, module)
    try:
        method = getattr(module, real_method_name)
    except AttributeError:
        return {
            "success": False,
            "error": (
                f"Method '{parts[1]}' not found on module "
                f"'{module_name}'."),
        }

    try:
        # Some OpenAI-compatible providers occasionally wrap function
        # arguments once more as {"parameters": {...}} or as a JSON string.
        # Unwrap only when the actual Python API has no parameter named
        # ``parameters``; genuine APIs using that name keep normal semantics.
        normalized_arguments = arguments
        if not isinstance(arguments, dict):
            return {"success": False, "error": "invalid_tool_arguments", "message": "arguments must be an object"}
        # Engine-installed adapters may have *args/**kwargs or no annotations.
        # Validate against the public API that generated the advertised schema,
        # not an implementation adapter's narrower or less precise signature.
        contract_method = _get_method_object(module_name, real_method_name) or method
        signature = inspect.signature(contract_method)
        if (set(arguments) == {"parameters"}
                and "parameters" not in signature.parameters):
            nested = arguments["parameters"]
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except json.JSONDecodeError:
                    nested = None
            if isinstance(nested, dict):
                normalized_arguments = nested
        # Validate before executing: malformed calls must not partially mutate
        # equipment state. Return the exact contract so the next turn can repair
        # its arguments instead of guessing enum spellings.
        signature.bind(**normalized_arguments)
        doc_params = _parse_docstring_params(inspect.getdoc(contract_method) or "")
        try:
            hints = get_type_hints(inspect.unwrap(contract_method))
        except (NameError, TypeError):
            hints = {}
        for name, value in normalized_arguments.items():
            param = signature.parameters[name]
            param = param.replace(annotation=hints.get(name, param.annotation))
            schema = _build_param_schema(name, param, doc_params.get(name))
            errors = list(Draft202012Validator(schema).iter_errors(value))
            if errors:
                return {"success": False, "error": "invalid_tool_arguments",
                        "parameter": name, "message": errors[0].message,
                        "expected": schema}
        result = method(**normalized_arguments)
        # Device modules historically used two result envelopes:
        # ``success: bool`` and ``status: success|error``.  Normalize them at
        # the dispatch boundary so the driver, audit log and passenger judge
        # cannot disagree about whether an operation actually succeeded.
        if isinstance(result, dict) and "success" not in result:
            status = str(result.get("status", "")).strip().lower()
            if "error" in result or status == "error":
                result = {**result, "success": False}
            elif status in {"success", "info"}:
                result = {**result, "success": True}
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Error calling {module_name}.{real_method_name}: {e}",
        }


def get_api_content_fc(modules: List[str]) -> str:
    """FC-mode variant of get_api_content.

    Formats method names as tool names (module__method) instead of vw.module.method,
    so the agent knows exactly which tool name to call.

    Args:
        modules: List of module names to look up.

    Returns:
        Formatted API documentation string with tool-name references.
    """
    apis_dict = utils.apis
    content = ""
    for module in modules:
        if module in apis_dict:
            content += f"\n--- {module.upper()} METHODS ---\n"
            for method_info in apis_dict[module]:
                tool_name = _canonical_tool_name(
                    module, method_info['name'])
                content += f"\n{tool_name}\n"
                content += "Description:\n"
                # Rewrite the source to remove 'self' param from def line
                source = method_info['source']
                # Replace "def method_name(self, ...)" → "def method_name(...)"
                # and "def method_name(self)" → "def method_name()"
                source = re.sub(r'def\s+(\w+)\s*\(self,\s*', r'def \1(', source)
                source = re.sub(r'def\s+(\w+)\s*\(self\)', r'def \1()', source)
                content += source + "\n"
    return content


def dispatch_discovery(vw, tool_name: str, arguments: dict, fc_mode: bool = False) -> Any:
    """Handle both discovery tools and regular module tools.

    Discovery tools:
      - get_modules → vw.get_modules()
      - get_module_API → vw.get_module_API(modules=[...]) or FC-mode variant

    Regular tools:
      - {module}__{method} → dispatch(vw, tool_name, arguments)

    Args:
        vw: VehicleWorld instance
        tool_name: Tool name string
        arguments: Tool arguments dict
        fc_mode: If True, get_module_API returns FC-formatted docs (tool names instead of vw.xxx)
    """
    if tool_name == "get_modules":
        return vw.get_modules()
    elif tool_name == "get_module_API":
        if fc_mode:
            requested = arguments.get("modules", [])
            unavailable = [
                module for module in requested
                if hasattr(vw, "has_module") and not vw.has_module(module)]
            if unavailable:
                return {
                    "success": False,
                    "error": "capability_not_available",
                    "modules": sorted(unavailable),
                }
            return get_api_content_fc(requested)
        else:
            return vw.get_module_API(**arguments)
    else:
        return dispatch(vw, tool_name, arguments)


# =====================================================================
# Lazy-loading discovery (3-level) for driving simulation
# =====================================================================

def generate_lazy_discovery_tools() -> List[Dict]:
    """Generate 2-level lazy discovery tools for driving simulation.

    Level 1: get_module_api  → one module's tool names + descriptions + intro
    Level 2: load_tools      → load specific tools by name (supports multiple)

    Module list is provided in the system prompt, so get_modules is not needed.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "get_module_api",
                "description": (
                    "Returns a brief introduction to a module and its available "
                    "API methods (names and one-line descriptions). "
                    "Use this to discover what a module offers, then load "
                    "only the ones you need. "
                    "Only accepts ONE module name at a time."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module": {
                            "type": "string",
                            "description": "A single module name, e.g. 'wiper'",
                        },
                    },
                    "required": ["module"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_tools",
                "description": (
                    "Load specific APIs by their full names so you can invoke them. "
                    "Names use the format 'module__method', e.g. "
                    "'wiper__carcontrol_wiperBlade_switch'. "
                    "You can load multiple in one call. "
                    "Discover exact names with get_module_api first; do not "
                    "infer a module's method names from other modules. "
                    "Only load what you actually need."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of API names to load, e.g. "
                                "['weather__weather_get_condition', "
                                "'wiper__carcontrol_wiperBlade_switch']"
                            ),
                        },
                    },
                    "required": ["tools"],
                },
            },
        },
    ]


# ── Module descriptions for system prompt and get_module_api ──
_MODULE_DESCRIPTIONS = {
    # Environment modules (query-only for agent)
    "weather":      "Weather conditions. Query rain, snow, fog, temperature, visibility, wind.",
    "dayNight":     "Day/night cycle. Query current time period (dawn/day/dusk/night) and light level.",
    "map":          "Static map, current location, road and POI queries. Live visual traffic is not returned as text.",
    "speedLimit":   "Speed limit. Query current speed limit zone.",
    "road_perception": (
        "Non-image sensor readings such as ego localization, weather and "
        "day/night. Camera-visible bodies, signals and crossings arrive only "
        "in the automatic driving image."),
    "frontRadar": (
        "Installed forward radar. Measures anonymous vehicle distance, "
        "lateral offset, relative speed and TTC; it does not identify road "
        "semantics."),
    "rearRadar": (
        "Installed rear radar. Measures anonymous vehicle distance, lateral "
        "offset, relative speed and TTC; it is not a rear camera."),
    "lidar": (
        "Optional scanning LiDAR. Its processed ego-centred top-down geometry "
        "image is automatically attached as LidarBEV at a vehicle wake. It "
        "does not show traffic-light state, vehicle-light effects, weather "
        "or day/night appearance, and exposes no text scan tool."),
    # Vehicle control modules
    "wiper":        "Windshield wipers. Turn on/off, set speed (low/medium/high).",
    "fogLight":     "Fog lights (front & rear). Turn on/off.",
    "lowBeamHeadlight": "Low beam headlights. Turn on/off.",
    "highBeamHeadlight": "High beam headlights. Turn on/off.",
    "positionLight": "Position/parking lights. Turn on/off.",
    "hazardLight":  "Hazard warning lights. Turn on/off.",
    "turnSignal":   "World-visible left/right turn indicators.",
    "horn":         "World-visible horn pulses with bounded duration and intensity.",
    "tailLight":    "Tail lights. Turn on/off.",
    "window":       "Windows (front-left/right, rear-left/right). Open/close.",
    "sunroof":      "Sunroof. Open/close.",
    "airConditioner": "Air conditioning. On/off, heating/cooling mode, temperature, fan speed, defog.",
    "seat":         "Seats. Heating, ventilation, position adjustment.",
    "door":         "Doors. Lock/unlock, open/close.",
    "HUD":          "Head-up display. Turn on/off, adjust brightness.",
    "readingLight": "Reading lights. Turn on/off.",
    "rearviewMirror": "Rearview mirrors. Fold/unfold, adjust angle.",
    "steeringWheel": "Steering wheel. Heating, position adjustment.",
    "sunshade":     "Sunshades. Open/close.",
    "trunk":        "Rear trunk. Open/close.",
    "frontTrunk":   "Front trunk (frunk). Open/close.",
    "fuelPort":     "Fuel port. Open/close.",
    "footPedal":    "Foot pedal. Adjust position.",
    # Infotainment modules
    "navigation":   "Driving & navigation. View suggested routes, explicitly select each junction maneuver, and submit continuous vehicle controls.",
    "music":        "Music player. Select available music sources and adjust volume.",
    "radio":        "Radio. Tune station, volume.",
    "video":        "Video player. Play/pause.",
    "bluetooth":    "Bluetooth. Connect/disconnect devices.",
    "conversation": "Phone call system. Make/answer/end calls.",
    "centerInformationDisplay": "Center display. Brightness, content settings.",
    "overheadScreen": "Overhead screen. On/off, content.",
    "instrumentPanel": "Instrument panel. Display settings.",
    # Broadcast module
    "broadcast":    "Driver announcement system. Broadcast traffic lights, road events, speed cameras, congestion, route info, and safety warnings to the driver.",
}


def get_module_list_for_prompt(vw=None, available_modules=None) -> str:
    """Return a formatted module list for inclusion in the system prompt.

    Groups modules by category for readability.
    """
    if available_modules is None and vw is not None:
        available_modules = vw.available_module_names()
    available = (
        set(available_modules)
        if available_modules is not None else None
    )
    if available is not None:
        # Virtual sensor APIs are engine capabilities, not cabin equipment.
        available.add("road_perception")
    lines = []
    lines.append("## Available Modules\n")
    if vw is not None and hasattr(vw, "capabilities"):
        chassis = vw.capabilities.chassis
        lines.append(
            f"Equipment profile: "
            f"`{vw.capabilities.equipment_profile}`. "
            "Only the modules listed below are installed; unavailable "
            "modules cannot be discovered or loaded.\n")
        lines.append(
            f"Chassis profile: `{vw.capabilities.chassis_profile}` "
            f"({chassis.length_m:.2f} m × {chassis.width_m:.2f} m, "
            f"max acceleration {chassis.max_acceleration_mps2:.2f} m/s², "
            f"max braking {chassis.max_braking_mps2:.2f} m/s²).\n")

    groups = [
        ("Environment (query-only)", ["road_perception", "weather", "dayNight", "map", "speedLimit"]),
        ("Physical sensors", ["frontRadar", "rearRadar", "lidar"]),
        ("Lights", ["fogLight", "lowBeamHeadlight", "highBeamHeadlight", "positionLight", "hazardLight", "turnSignal", "tailLight", "readingLight", "HUD"]),
        ("Road communication", ["horn"]),
        ("Body", ["wiper", "window", "sunroof", "door", "trunk", "frontTrunk", "fuelPort", "sunshade", "rearviewMirror"]),
        ("Comfort", ["airConditioner", "seat", "steeringWheel", "footPedal"]),
        ("Driving", ["navigation"]),
        ("Infotainment", ["music", "radio", "video", "bluetooth", "conversation", "centerInformationDisplay", "overheadScreen", "instrumentPanel"]),
        ("Broadcast", ["broadcast"]),
    ]
    listed = set()
    for group_name, module_names in groups:
        installed = [
            name for name in module_names
            if available is None or name in available]
        if not installed:
            continue
        listed.update(installed)
        lines.append(f"**{group_name}**")
        for m in installed:
            desc = _MODULE_DESCRIPTIONS.get(m, "")
            lines.append(f"  - `{m}`: {desc}")
        lines.append("")
    if available is not None:
        remaining = sorted(available - listed)
        if remaining:
            lines.append("**Other installed modules**")
            for module in remaining:
                lines.append(
                    f"  - `{module}`: "
                    f"{_MODULE_DESCRIPTIONS.get(module, '')}")
            lines.append("")

    return "\n".join(lines)


def get_brief_module_api(
    module_name: str, available_modules=None, denied_tool_names=None,
):
    """Return a brief introduction + tool list for one module.

    Includes a module-level description, then each method's full tool
    name and a one-line summary.

    Args:
        module_name: Single module name (e.g. 'wiper').

    Returns:
        Brief summary string, or error message if module not found.
    """
    available = (
        set(available_modules)
        if available_modules is not None else None)
    denied = set(denied_tool_names or ())
    if module_name == "road_perception":
        from simulation.perception_tools import VEHICLE_PERCEPTION_TOOLS
        visible_schemas = [
            schema for schema in VEHICLE_PERCEPTION_TOOLS
            if schema["function"]["name"] not in denied]
        lines = [
            "Module: road_perception — "
            + _MODULE_DESCRIPTIONS["road_perception"],
            f"{len(visible_schemas)} APIs available:",
        ]
        for schema in visible_schemas:
            function = schema["function"]
            lines.append(
                f"  - {function['name']}: {function['description']}")
        lines.extend(["", "NOTE: Load only the observations you need."])
        return "\n".join(lines)
    if available is not None and module_name not in available:
        return {
            "success": False,
            "error": "capability_not_available",
            "module": module_name,
        }

    apis_dict = utils.apis
    if module_name not in apis_dict:
        return f"Module '{module_name}' not found."

    methods = apis_dict[module_name]

    # Module intro
    intro = _MODULE_DESCRIPTIONS.get(module_name, "")
    lines = [f"Module: {module_name} — {intro}"]
    visible_methods = [
        method for method in methods
        if (_make_tool_name(module_name, method["name"]) not in denied
            and _canonical_tool_name(
                module_name, method["name"]) not in denied)]
    lines.append(f"{len(visible_methods)} APIs available:")
    for m in visible_methods:
        tool_name = _canonical_tool_name(module_name, m['name'])
        # Extract first line of docstring
        source = m["source"]
        doc_match = re.search(r'"""(.+?)"""', source, re.DOTALL)
        if doc_match:
            first_line = doc_match.group(1).strip().split("\n")[0].strip()
        else:
            first_line = m["name"]
        lines.append(f"  - {tool_name}: {first_line}")
    lines.append("")
    lines.append(
        "NOTE: Only use the ones you actually need."
    )
    return "\n".join(lines)


def generate_skill_tool(skill_loader) -> Dict:
    """Generate the load_skill tool schema from a SkillLoader instance.

    The enum is dynamically populated from discovered skill files,
    so user-defined skills automatically appear.
    """
    skill_names = [s.name for s in skill_loader.list_skills()]
    return {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load a driving skill guide by name. Returns detailed "
                "step-by-step rules for handling a specific situation. "
                "Once loaded, the rules persist in your context — "
                "do not re-load it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "enum": skill_names,
                        "description": "Name of the skill to load",
                    },
                },
                "required": ["skill_name"],
            },
        },
    }


def dispatch_lazy_discovery(vw, tool_name: str, arguments: dict,
                            tools_list: List[Dict],
                            skill_loader=None,
                            extra_schemas: Optional[List[Dict]] = None,
                            virtual_module_catalogs: Optional[
                                Dict[str, str]] = None,
                            denied_tool_names: Optional[set] = None) -> Any:
    """Handle 2-level lazy discovery tools + load_skill + regular module tools.

    Args:
        vw:           VehicleWorld instance.
        tool_name:    Tool name string.
        arguments:    Tool arguments dict.
        tools_list:   The *live* tools list; load_tools mutates it in-place.
        skill_loader: Optional SkillLoader instance for load_skill dispatch.

    Returns:
        Result of the tool call.
    """
    denied = set(denied_tool_names or ())
    if tool_name in denied:
        return {
            "success": False,
            "error": "visual_information_is_image_only",
            "tool": tool_name,
        }

    if tool_name == "load_skill":
        if skill_loader is None:
            return {"success": False, "error": "Skill loader not available."}
        available = (
            vw.available_module_names()
            if hasattr(vw, "available_module_names") else None)
        return skill_loader.load_skill(
            arguments.get("skill_name", ""),
            available_modules=available,
        )

    elif tool_name == "get_module_api":
        module = arguments.get("module", "")
        if module in (virtual_module_catalogs or {}):
            return virtual_module_catalogs[module]
        available = (
            vw.available_module_names()
            if hasattr(vw, "available_module_names") else None)
        return get_brief_module_api(
            module, available, denied_tool_names=denied)

    elif tool_name == "load_tools":
        requested = arguments.get("tools", [])
        if (not isinstance(requested, list)
                or any(not isinstance(name, str) for name in requested)):
            return {"success": False, "error": "invalid_tool_arguments",
                    "parameter": "tools", "expected": "array of strings"}
        if not requested:
            return {"success": False, "error": "No tools specified."}
        denied_requested = sorted(set(requested) & denied)
        if denied_requested:
            return {
                "success": False,
                "error": "visual_information_is_image_only",
                "tools": denied_requested,
            }

        # Intercept attempts to load meta-tools that are already pre-loaded
        _PRELOADED = {
            "get_module_api", "load_tools", "load_skill", "todo_manage",
            "finish"}
        preloaded_hits = [t for t in requested if t in _PRELOADED]
        if preloaded_hits:
            remaining = [t for t in requested if t not in _PRELOADED]
            if not remaining:
                return {
                    "success": False,
                    "error": "tools_already_loaded_call_directly",
                    "tools": preloaded_hits,
                }
            requested = remaining

        # Road-perception tools are lazy and use canonical module__method
        # names.
        from simulation.perception_tools import (
            VEHICLE_PERCEPTION_TOOLS, VEHICLE_PERCEPTION_TOOL_NAMES)
        extra_by_name = {
            item["function"]["name"]: item
            for item in (extra_schemas or [])}
        module_requested = [
            name for name in requested
            if (name not in VEHICLE_PERCEPTION_TOOL_NAMES
                and name not in extra_by_name)]

        # Collect unique modules from requested cabin tool names.
        modules_needed = set()
        for t in module_requested:
            parts = t.split("__", 1)
            if len(parts) == 2:
                modules_needed.add(parts[0])
        unavailable = sorted(
            module for module in modules_needed
            if hasattr(vw, "has_module") and not vw.has_module(module))
        if unavailable:
            return {
                "success": False,
                "error": "capability_not_available",
                "modules": unavailable,
            }

        # Generate schemas for those modules, then filter to only requested
        all_schemas = (
            generate_tools_schema(modules=list(modules_needed))
            if modules_needed else [])
        module_by_name = {
            item["function"]["name"]: item for item in all_schemas}
        canonical_names = set()
        # Keep legacy schemas loadable, while also exposing compact aliases
        # backed by the exact same parameter contract.
        for module in modules_needed:
            for method in utils.apis.get(module, []):
                legacy = _make_tool_name(module, method["name"])
                alias = _canonical_tool_name(module, method["name"])
                if alias == legacy or legacy not in module_by_name:
                    continue
                alias_schema = copy.deepcopy(module_by_name[legacy])
                alias_schema["function"]["name"] = alias
                module_by_name[alias] = alias_schema
                canonical_names.add(alias)
        perception_by_name = {
            item["function"]["name"]: item
            for item in VEHICLE_PERCEPTION_TOOLS}
        existing_names = {
            item["function"]["name"] for item in tools_list}
        valid_names = (
            set(module_by_name) | VEHICLE_PERCEPTION_TOOL_NAMES
            | set(extra_by_name)) - denied
        invalid = [
            name for name in requested
            if name not in valid_names and name not in existing_names]
        if invalid:
            # Suggest only canonical, permitted names from this vehicle's
            # catalog. Never silently dispatch a guessed replacement, and
            # keep mixed valid/invalid loads transactional.
            suggestions = {}
            for name in invalid:
                module = name.split("__", 1)[0]
                method = name.split("__", 1)[-1]
                # Common hallucination in the sample: copying another
                # module's carcontrol_<module>_ prefix onto a short API.
                method = method.removeprefix(f"carcontrol_{module}_")
                candidates = sorted(candidate for candidate in valid_names
                                    if candidate.startswith(module + "__"))
                suggestions[name] = sorted(
                    candidates, key=lambda candidate: (
                        -difflib.SequenceMatcher(None, method, candidate.split("__", 1)[-1]
                            .removeprefix(f"carcontrol_{module}_")).ratio(),
                        candidate not in canonical_names,
                        candidate))[:8]
            return {
                "success": False, "error": f"Unknown: {invalid}",
                "suggested_tools": suggestions,
                "next_action": "Call get_module_api with the module name "
                               "for its full catalog, then load exact names.",
            }

        # Loading is transactional: a mixed valid/invalid request never
        # mutates the live tool set before returning an error.
        added = []
        for name in requested:
            if name in existing_names:
                continue
            schema = (perception_by_name.get(name)
                      or module_by_name.get(name)
                      or extra_by_name.get(name))
            if schema is None:
                continue
            tools_list.append(copy.deepcopy(schema))
            existing_names.add(name)
            added.append(name)
        if added:
            loaded_schemas = [
                item for item in tools_list
                if item.get("function", {}).get("name") in set(added)]
            schema_hash = hashlib.sha256(json.dumps(
                loaded_schemas, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest()
            return {
                "success": True,
                "loaded": added,
                "count": len(added),
                "schema_sha256": schema_hash,
            }
        return {
            "success": False,
            "error": "tools_already_loaded_call_directly",
            "tools": requested,
        }

    else:
        return dispatch(vw, tool_name, arguments)
