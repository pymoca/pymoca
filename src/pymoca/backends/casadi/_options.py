def _get_default_options():
    """
    Returns a dictionary with the available compiler options and the default values

    Returns:
        default_options (Dict)

    """
    return {
        'library_folders': [],
        'verbose': False,
        'check_balanced': True,
        'mtime_check': True,
        'cache': False,
        'codegen': False,
        'expand_mx': False,
        'unroll_loops': True,
        'inline_functions': True,
        'expand_vectors': False,
        'replace_parameter_expressions': False,
        'replace_constant_expressions': False,
        'eliminate_constant_assignments': False,
        'replace_parameter_values': False,
        'replace_constant_values': False,
        'eliminable_variable_expression': None,
        'factor_and_simplify_equations': False,
        'detect_aliases': False,
        'reduce_affine_expression': False,
    }


def _merge_default_options(options):
    if options is None:
        return _get_default_options()
    elif isinstance(options, dict):
        default_options = _get_default_options()
        default_options.update(options)
        return default_options
    else:
        raise TypeError('options must be of type dict')
