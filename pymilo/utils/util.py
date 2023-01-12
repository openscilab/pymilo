from numpy import ndarray


def get_sklearn_type(model):
    """
    Return sklearn model type.

    :param model: sklearn model
    :type model: any sklearn's model class
    :return: model type as str
    """
    raw_type = type(model)
    return str(raw_type).split(".")[-1][:-2]


def is_primitive(obj):
    """
    Check if the given object is primitive.

    :param obj: given object
    :type obj: any valid type
    :return: True if object is primitive
    """
    return not hasattr(obj, '__dict__')


def is_iterable(obj):
    """
    Check if the given object is iterable.

    :param obj: given object
    :type obj: any valid type
    :return: True if object is iterable
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def check_str_in_iterable(field, content):
    """
    TODO: Complete docstring.
    """
    if not is_iterable(content):
        return False
    if isinstance(content, ndarray):
        # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur.
        return False
    else:
        return field in content
