
#######################################################################################################
# modules
#######################################################################################################

# A module is a .py file containing Python definitions and statements. The module name is the file name without suffix .py. 
# Within a module, the module’s name (as a string) is available as the value of the global variable __name__. (another global variable __file__ is the pathname of the file of the module) 

# A module can contain executable statements as well as function definitions. These statements are intended to initialize the module. 
# Statements are executed only the first time the module name is encountered in an import statement. 
# Each module has its own private symbol table (as the module's __dict__), which is used as the global symbol table by all functions defined in that module.  
# A module is loaded only once. To reload a module, use  import importlib; importlib.reload(modulename)

# Modules can import other modules using the import MODULE_NAME command.  
# The imported module names are placed in the importing module’s global symbol table.  
# There is a variant of the import statement that imports names from a module directly into the importing module’s symbol table: from MODULE_NAME import NAMES. 
# This does NOT introduce the module name from which the imports are taken in the local symbol table.


# Note that when using syntax from package import item, the item can be either a submodule (or subpackage) of the package, 
# or some other name defined in the package ('s __init__.py), like a function, class or variable. 
# The import statement first tests whether the item is defined in the package __init__.py file; 
# if not, it assumes it is a module and attempts to load it under the directory of package. If it fails to find it, an ImportError exception is raised.

# Contrarily, when using syntax like import item.subitem.subsubitem, each item except for the last must be a package; 
# the last item can be a module or a package but can’t be a class or function or variable defined in the previous item. 
# To import function, class or variable, use from … import … syntax.

# The import statement uses the following convention: 
# if a package’s __init__.py code defines a list named __all__, it is taken to be the list of module names that should be imported when from package import * is encountered. 
# If __all__ is not defined, it only imports any names defined (and submodules explicitly loaded) by __init__.py. 


# Note that when running a Python script, sys.path doesn’t care what your current “working directory” or pwd is. 
# It only cares about the path to the python script being run. 
# For example, if my shell is currently in folder test/  and I run python ./packA/subA/subA1.py, then sys.path includes test/packA/subA/ but NOT test/.


# An absolute import uses the full path (not starting from disk root, but starting from the project’s root folder) to the desired module to import. 
# Recall that a project's root folder is the path of the script invoked on the command line and is agnostic to the working directory on the command line.


#######################################################################################################
# RE
#######################################################################################################

# https://docs.python.org/3.8/howto/regex.html#regex-howto
#  re.compile(pattern, flags=0) Compile a regular expression pattern into a regular expression object, which can be used for matching using its match(), search() and other methods.
#     The expression’s behaviour can be modified by specifying a flags value, combined using bitwise OR (the | operator).

# Use Python’s raw string notation for writing RE patterns; backslashes are not handled in any special way in a string literal prefixed with 'r', 
# so r"\n" is a two-character string containing '\' and 'n', while "\n" is a one-character string containing a newline. 

# Once you have an object representing a compiled regular expression, you can use its following methods.
# match() : Determine if the RE matches at the start of the string. Return the first matched string.
# search() : Scan through a string, looking for any location where this RE matches. Return the first matched string.
# findall() : Find all substrings where the RE matches, and returns them as a list.
# finditer() : Find all substrings where the RE matches, and returns them as an iterator.

# The match() function only checks if the RE matches at the beginning of the string while search() will scan forward through the string for a match. 
# Remember, match() will only report a successful match which will start at 0; if the match wouldn’t start at zero, match() will not report it.

# match() and search() return None if no match can be found. 
# If they’re successful, a match object instance is returned, containing information about the match: where it starts and ends, the substring it matched, and more.

# Match object instances also have several methods and attributes; the most important ones are:
# group() : Return the string matched by the RE
# start() : Return the starting position of the match
#           Since the match() method only checks if the RE matches at the start of a string, start() will always be zero if if there is a match. 
# end() : Return the ending position of the match
# span() : Return a tuple containing the (start, end) positions of the match



# Regular expressions can contain both special and ordinary characters. 
# Most ordinary characters simply match themselves. You can concatenate ordinary characters.
# Some characters, like '|' or '(', are special. Special characters either stand for classes of ordinary characters, or affect how the regular expressions around them are interpreted.
# Complete list of the special characters: . ^ $ * + ? { } [ ] \ | ( )

# Repetition qualifiers (*, +, ?, {m,n}, etc) cannot be directly nested.  
# To apply a second repetition to an inner repetition, parentheses may be used. For example, the expression (?:a{6})* matches any multiple of six 'a' characters.

# The special characters are:
# .
#     (Dot.) In the default mode, this matches any single character except a newline. If the DOTALL flag has been specified, this matches any character including a newline.
# ^
#     (Caret.) Matches the start of the string, and in MULTILINE mode also matches immediately after each newline.
# $
#     Matches the end of the string or just before the newline at the end of the string, and in MULTILINE mode also matches before a newline. 
# *
#     Causes the resulting RE to match 0 or more repetitions of the preceding RE, as many repetitions as are possible. ab* will match ‘a’, ‘ab’, or ‘a’ followed by any number of ‘b’s.
# +
#     Causes the resulting RE to match 1 or more repetitions of the preceding RE. ab+ will match ‘a’ followed by any non-zero number of ‘b’s; it will not match just ‘a’.
# ?
#     Causes the resulting RE to match 0 or 1 repetitions of the preceding RE. ab? will match either ‘a’ or ‘ab’.
# *?, +?, ??
#     The '*', '+', and '?' qualifiers are all greedy, which means they match as much text as possible. 
#     Sometimes this behaviour isn’t desired; if the RE <.*> is matched against '<a> b <c>', it will match the entire string, and not just '<a>'. 
#     Adding ? after the qualifier makes it perform the match in non-greedy or minimal fashion; as few characters as possible will be matched. 
#     Using the RE <.*?> will match only '<a>'.
# {m}
#     Specifies that exactly m copies of the previous RE should be matched; fewer matches cause the entire RE not to match. 
#     For example, a{6} will match exactly six 'a' characters, but not five.
# {m,n}
#     Causes the resulting RE to match from m to n repetitions of the preceding RE, attempting to match as many repetitions as possible. 
#     For example, a{3,5} will match from 3 to 5 'a' characters. Omitting m specifies a lower bound of zero, and omitting n specifies an infinite upper bound. 
#     As an example, a{4,}b will match 'aaaab' or a thousand 'a' characters followed by a 'b', but not 'aaab'. 
# {m,n}?
#     Causes the resulting RE to match from m to n repetitions of the preceding RE, attempting to match as few repetitions as possible. 
#     For example, on the 6-character string 'aaaaaa', a{3,5} will match 5 'a' characters, while a{3,5}? will only match 3 characters.
# \
#     Either escapes special characters (permitting you to match characters like '*', '?', and so forth), or signals a special sequence; special sequences are discussed below.
#     If you’re not using a raw string to express the pattern, remember that Python also uses the backslash as an escape sequence in string literals; 
#     if the escape sequence isn’t recognized by Python’s parser, the backslash and subsequent character are included in the resulting string. 
# []
#     Used to indicate a set of characters. In a set:
#         Characters can be listed individually, e.g. [amk] will match 'a', 'm', or 'k'.
#         Ranges of characters can be indicated by giving two characters and separating them by a '-', for example [a-z] will match any lowercase ASCII letter, 
#         [0-5][0-9] will match all the two-digits numbers from 00 to 59, and [0-9A-Fa-f] will match any hexadecimal digit. 
#         If - is escaped (e.g. [a\-z]) or if it’s placed as the first or last character (e.g. [-a] or [a-]), it will match a literal '-'.
#         Special characters lose their special meaning inside sets. For example, [(+*)] will match any of the literal characters '(', '+', '*', or ')'.
#         Character classes such as \w or \S are also accepted inside a set, although the characters they match depends on whether ASCII or LOCALE mode is in force.
#         Characters that are not within a range can be matched by complementing the set. If the first character of the set is '^', all the characters that are not in the set will be matched. 
#         For example, [^5] will match any character except '5', and [^^] will match any character except '^'. ^ has no special meaning if it’s not the first character in the set.
#         To match a literal ']' inside a set, precede it with a backslash, or place it at the beginning of the set. For example, both [()[\]{}] and []()[{}] will both match a parenthesis.
# |
#     A|B, where A and B can be arbitrary REs, creates a regular expression that will match either A or B. An arbitrary number of REs can be separated by the '|' in this way. 
#     This can be used inside groups as well. As the target string is scanned, REs separated by '|' are tried from left to right. 
#     When one pattern completely matches, that branch is accepted. This means that once A matches, B will not be tested further, even if it would produce a longer overall match. 
#     In other words, the '|' operator is never greedy. To match a literal '|', use \|, or enclose it inside a character class, as in [|].

# (...)
#     Matches whatever regular expression is inside the parentheses, and indicates the start and end of a group; 
#     the contents of a group can be retrieved after a match has been performed, and can be matched later in the string with the \number special sequence. 
#     To match the literals '(' or ')', use \( or \), or enclose them inside a character class: [(], [)].

# (?:...)
#     A non-capturing version of regular parentheses. Matches whatever regular expression is inside the parentheses, but the substring matched by the group cannot be retrieved after performing a match or referenced later in the pattern.
# (?P<name>...)
#     Similar to regular parentheses, but the substring matched by the group is accessible via the symbolic group name name. 
# (?P=name)
#     A backreference to a named group; it matches whatever text was matched by the earlier group named name.

# \d
#     Matches any decimal digit; this is equivalent to the class [0-9].
# \D
#     Matches any non-digit character; this is equivalent to the class [^0-9].
# \s
#     Matches any whitespace character; this is equivalent to the class [ \t\n\r\f\v].
# \S
#     Matches any non-whitespace character; this is equivalent to the class [^ \t\n\r\f\v].
# \w
#     Matches any alphanumeric character; this is equivalent to the class [a-zA-Z0-9_].
# \W
#     Matches any non-alphanumeric character; this is equivalent to the class [^a-zA-Z0-9_].




#######################################################################################################
# dataclass
#######################################################################################################

# @dataclasses.dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
#   - This function is a decorator that is used to add generated special methods to classes, as described below.
#   - The dataclass() decorator examines the class to find fields. 
#     !! A field is defined as class variable that has a type annotation. A field can be either normal python syntax, or defined as field() !!
#   - The order of the fields in all of the generated methods is the order in which they appear in the class definition.
#   - The dataclass() decorator will add various “dunder” methods to the class, described below. 
#     If any of the added methods already exist on the class, the behavior depends on the parameter, as documented below. 
#     The decorator returns the same class that is called on; no new class is created.
#   - If dataclass() is used just as a simple decorator with no parameters, it acts as if it has the default values documented in this signature. 
# 

# The parameters to dataclass() are:
#   - init: If true (the default), a __init__() method will be generated. If the class already defines __init__(), this parameter is ignored.
#     -- The generated __init__() code will call a method named __post_init__(), if __post_init__() is defined on the class. 
#     -- If any InitVar fields are defined, they will also be passed to __post_init__() in the order they were defined in the class. 
#     -- If no __init__() method is generated, then __post_init__() will not automatically be called.
#   - repr: If true (the default), a __repr__() method will be generated. The generated repr string will have the class name and the name and repr of each field, in the order they are defined in the class. 
#     If the class already defines __repr__(), this parameter is ignored.
#   - eq: If true (the default), an __eq__() method will be generated. This method compares the class as if it were a tuple of its fields, in order. Both instances in the comparison must be of the identical type.
#     If the class already defines __eq__(), this parameter is ignored.
#   - order: If true (the default is False), __lt__(), __le__(), __gt__(), and __ge__() methods will be generated. These compare the class as if it were a tuple of its fields, in order. 
#     Both instances in the comparison must be of the identical type. If order is true and eq is false, a ValueError is raised.
#     If the class already defines any of __lt__(), __le__(), __gt__(), or __ge__(), then TypeError is raised.
#   - unsafe_hash: If False (the default), a __hash__() method is generated according to how eq and frozen are set.
#     __hash__() is used by built-in hash(), and when objects are added to hashed collections such as dictionaries and sets. 
#     Having a __hash__() implies that instances of the class are immutable. 
#     By default, dataclass() will not implicitly add a __hash__() method unless it is safe to do so. Neither will it add or change an existing explicitly defined __hash__() method. 
#     Setting the class attribute __hash__ = None has a specific meaning to Python, as described in the __hash__() documentation.
#     If __hash__() is not explicit defined, or if it is set to None, then dataclass() may add an implicit __hash__() method. 
#     Although not recommended, you can force dataclass() to create a __hash__() method with unsafe_hash=True. This might be the case if your class is logically immutable but can nonetheless be mutated. This is a specialized use case and should be considered carefully.
#     Here are the rules governing implicit creation of a __hash__() method. Note that you cannot both have an explicit __hash__() method in your dataclass and set unsafe_hash=True; this will result in a TypeError.
#     -- If eq and frozen are both true, by default dataclass() will generate a __hash__() method for you. 
#     -- If eq is true and frozen is false, __hash__() will be set to None, marking it unhashable (which it is, since it is mutable). 
#     -- If eq is false, __hash__() will be left untouched meaning the __hash__() method of the superclass will be used (if the superclass is object, this means it will fall back to id-based hashing).
#   - frozen: If true (the default is False), assigning to fields will generate an exception. This emulates read-only frozen instances. 



# fields may optionally specify a default value, using normal Python syntax:
# @dataclass
# class C:
#     a: int       # 'a' has no default value
#     b: int = 0   # assign a default value for 'b'
# In this example, both a and b will be included in the added __init__() method, which will be defined as:
# def __init__(self, a: int, b: int = 0):
#     self.a = a
#     self.b = b
# TypeError will be raised if a field without a default value follows a field with a default value. 
# In addition, since b has default value, b is also added to class C as class attribute b = 0. You can access it without any instance of class C, using C.b. But not very useful.
# But a is not added as class attribute since it does not have default value.

# fields can also be defined by a call to the provided field() function to provide additional information. For example:
# dataclasses.field(*, default=MISSING, default_factory=MISSING, repr=True, hash=None, init=True, compare=True, metadata=None)
# @dataclass
# class C:
#     mylist: List[int] = field(default_factory=list)

# c = C()
# c.mylist += [1, 2, 3]

# As shown above, the MISSING value is a sentinel object used to detect if the default and default_factory parameters are provided. 
# This sentinel is used because None is a valid value for default. 
# No code should directly use the MISSING value.

# The parameters to field() are:
#   - default: If provided, this will be the default value for this field. 
#   - default_factory: If provided, it must be a zero-argument callable that will be called when a default value is needed for this field.  
#     It is an error to specify both default and default_factory.
#   - init: If true (the default), this field is included as a parameter to the generated __init__() method.
#   - repr: If true (the default), this field is included in the string returned by the generated __repr__() method.
#   - compare: If true (the default), this field is included in the generated equality and comparison methods (__eq__(), __gt__(), et al.).
#   - hash: This can be a bool or None. If true, this field is included in the generated __hash__() method. If None (the default), use the value of compare: this would normally be the expected behavior. 
#     A field should be considered in the hash if it’s used for comparisons. Setting this value to anything other than None is discouraged.
#   - metadata: This can be a mapping/dict or None. None is treated as an empty dict. 
#     This value is wrapped in MappingProxyType() to make it read-only, and exposed on the Field object. 


#  dataclasses.asdict(instance, *, dict_factory=dict) Converts the dataclass instance to a dict (by using the factory function dict_factory). 
# Each dataclass is converted to a dict of its fields, as name: value pairs. dataclasses, dicts, lists, and tuples are recursed into. 



#######################################################################################################
# decorator
#######################################################################################################



