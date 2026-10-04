import inspect
import json
import os
import shutil
import re
import functools
import ast
import threading
import queue
from RestrictedPython import compile_restricted_exec

# Create a global dictionary to store decorated methods, grouped by type
apis = {}

def api(module_type="default"):
    def decorator(func):
        # Get the source code of the function
        source_code = inspect.getsource(func)

        # Extract function header (def line)
        header_pattern = r"(def\s+\w+\s*\([^)]*\)[^:]*:)"
        header_match = re.search(header_pattern, source_code)
        function_header = header_match.group(1) if header_match else "def unknown():"

        # Extract docstring if it exists
        docstring = func.__doc__

        # Combine header and docstring
        extracted_info = function_header
        if docstring:
            # Format docstring with proper indentation
            docstring_lines = docstring.strip().split('\n')
            indented_docstring = '\n    '.join([line.strip() for line in docstring_lines])
            extracted_info += '\n    """\n    ' + indented_docstring + '\n    """'

        # Record the decorated method by type
        if module_type not in apis:
            apis[module_type] = []

        # Store only the function header and docstring
        apis[module_type].append({
            'name': func.__name__,
            'source': extracted_info
        })

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # ── Coupling Layer: Pre-execution constraint check ──
            try:
                module_instance = args[0] if args else None
                engine = getattr(module_instance, '_constraint_engine', None)
                if engine is None:
                    return {
                        "success": False,
                        "error": "Module is not attached to a VehicleWorld",
                    }
                can_execute, constraint_results = engine.check_pre(
                    module_type, func.__name__, module_instance, args, kwargs
                )

                if not can_execute:
                    # HARD constraint blocked execution
                    error_messages = [
                        f"[{r.constraint_name}] {r.message}"
                        for r in constraint_results
                        if r.level.value == "hard"
                    ]
                    return {
                        "success": False,
                        "error": "Operation blocked by safety constraint",
                        "constraints": error_messages
                    }

                # Collect SOFT/ADVISORY warnings to attach to result
                warnings = [
                    f"[{r.constraint_name}] {r.message}"
                    for r in constraint_results
                ]
            except ImportError:
                warnings = []
            except Exception:
                warnings = []

            # ── Execute the original function ──
            result = func(*args, **kwargs)

            # ── Coupling Layer: Post-execution constraint check ──
            try:
                module_instance = args[0] if args else None
                engine = getattr(module_instance, '_constraint_engine', None)
                if engine is None:
                    return {
                        "success": False,
                        "error": "Module is not attached to a VehicleWorld",
                    }
                post_results = engine.check_post(
                    module_type, func.__name__, module_instance, args, kwargs, result
                )
                post_warnings = [
                    f"[{r.constraint_name}] {r.message}"
                    for r in post_results
                ]
                warnings.extend(post_warnings)
            except ImportError:
                pass
            except Exception:
                pass

            # ── Attach warnings to result if any ──
            if warnings and isinstance(result, dict):
                result["_coupling_warnings"] = warnings

            return result

        return wrapper

    # Support two usage methods:
    # @api  or  @api("type")
    if callable(module_type):
        func = module_type
        module_type = "default"
        return decorator(func)

    return decorator

def save_json_file(data, output_file):
    """
    Save data as a JSON file
    :param data: Data to be saved
    :param output_file: Output file path
    """
    dir_name = os.path.dirname(output_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_api_content(modules):
    apis_content = ""
    for module in modules:
        if module in apis:
            apis_content += f"\n--- {module.upper()} METHODS ---\n"
            for method_info in apis[module]:
                apis_content += f"\nMethod name: vw.{module}.{method_info['name']}\n"
                apis_content += "API description:\n"
                apis_content += method_info['source'] + "\n"
            apis_content += f"\nMethod name: vw.{module}.to_dict()\n"
            apis_content += f"API description:\nGet the current {module} system status information."
    return apis_content


def delete_all_contents(folder_path):
    """
    Delete all files and folders in the specified directory.
    
    Args:
        folder_path (str): Path to the folder to clean
        
    Returns:
        tuple: (success, message) with operation status and details
    """
    try:
        # Check if the path exists
        if not os.path.exists(folder_path):
            return False, f"Error: The path '{folder_path}' does not exist."
        
        # Check if the path is a directory
        if not os.path.isdir(folder_path):
            return False, f"Error: '{folder_path}' is not a directory."
        
        file_count = 0
        dir_count = 0
        
        # List all items in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # Handle files
            if os.path.isfile(item_path):
                os.remove(item_path)
                file_count += 1
                print(f"Deleted file: {item_path}")
            
            # Handle directories
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                dir_count += 1
                print(f"Deleted directory: {item_path}")
        
        return True, f"Successfully deleted {file_count} files and {dir_count} directories from '{folder_path}'."
    
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

def execute(code_str: str, local_vars: dict = None, global_vars: dict = None, timeout: int = 10) -> str:
    if local_vars is None:
        local_vars = {}
    if global_vars is None:
        global_vars = globals()

    output_lines = []
    result_queue = queue.Queue()

    def custom_print(*args, **kwargs):
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        output_lines.append(sep.join(str(arg) for arg in args) + end)

    local_vars['print'] = custom_print

    def exec_thread():
        try:
            modified_code = add_prints_to_function_calls(code_str)
            safe_code = compile_restricted_exec(modified_code)
            if safe_code:
                print(modified_code)
                exec(modified_code, global_vars, local_vars)
                output = ''.join(output_lines)
                result_queue.put(output if output else "Code executed successfully, no output.")
            else:
                result_queue.put("Compilation failed")
        except Exception:
            result_queue.put(f"Execution error:\n{traceback.format_exc()}")

    thread = threading.Thread(
        target=exec_thread
    )
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return f"Execution timeout (exceeded {timeout} seconds)"
    try:
        output = result_queue.get(block=False)
        return output
    except queue.Empty:
        return "Execution terminated abnormally, no return result"
     
def add_prints_to_function_calls(code_str):
    """
    Use AST parsing to extract independent function calls and add print statements
    
    Args:
        code_str: Original code string
        
    Returns:
        Modified code string with print statements
    """
    # Parse code into AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        # Return original code if there are syntax errors
        return code_str
    
    # Find independent function calls (function calls in top-level expression statements)
    calls_to_instrument = []
    
    class FunctionCallFinder(ast.NodeVisitor):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call):
                # Ensure it's not a print call
                if not (isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'):
                    calls_to_instrument.append((node.lineno, node.col_offset))
            self.generic_visit(node)
    
    FunctionCallFinder().visit(tree)
    
    # Split code lines and add print statements
    lines = code_str.split('\n')
    modified_lines = []
    
    # Line numbers start from 1
    for i, line in enumerate(lines, 1):
        modified_lines.append(line)
        
        # Check if current line contains function calls that need print statements
        for lineno, col_offset in calls_to_instrument:
            if lineno == i:
                # Extract indentation
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                
                # Extract function call
                stripped_line = line.strip()
                
                # Save result to variable
                result_line = indent + "_result = " + stripped_line
                # Use variable method to print function call content, completely avoiding quote issues in strings
                print_line = indent + "func_call = " + repr(stripped_line) + "\n" + indent + "print(\"Called:\", func_call, \"Return value:\", repr(_result))"

                # Replace original line with new code
                modified_lines[-1] = result_line + "\n" + print_line
                break
    
    return '\n'.join(modified_lines)

def capitalize_first(s):
    if not s:
        return s
    return s[0].upper() + s[1:]
