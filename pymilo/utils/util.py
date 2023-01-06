from numpy import ndarray


def get_sklearn_type(model):
    raw_type = type(model)
    return str(raw_type).split(".")[-1][:-2]


def is_primitive(obj):
    return not hasattr(obj, '__dict__')


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def check_str_in_iterable(field, content):
    if (not (is_iterable(content))):
        return False
    if (isinstance(content, ndarray)):
        # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur#:~:text=There%20is%20a,adopt%20Numpy%20style.
        return False
    else:
        return field in content
