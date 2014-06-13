import enum


class ClassType(enum.Enum):
    generic = 0
    model = 1
    record = 2
    operator_record = 3
    block = 4
    connector = 5
    operator_connector = 6
    type = 7
    package = 8
    function = 9


class Access(enum.Enum):
    private = 0
    public = 1
    protected = 2


class StoredDefinition(object):

    def __init__(self, ast):
        pass


class Class(object):

    def __init__(self):

        # defaults
        self.encapsulated = False
        self.partial = False
        self.type = ClassType.generic
        self.comment = ""
        self.name = ""
        self.composition = None
        self.array_subscripts = None
        self.enumeration = None
        self.language_specification = None
        self.external_function_call = None
        self.external_function_call_annotation = None


class Element(object):

    def __init__(self, ast):
        self.access = Access.private
