import importlib

def get_func(func_name):
    """
        Helper to return a function object by name. func_name must identify 
        a function in this module or the path to a function relative to the base
        module.
        @ func_name: function name.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except:
        raise RuntimeError(f'Failed to find function: {func_name}')
