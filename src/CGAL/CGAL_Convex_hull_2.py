# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

"""SWIG wrapper for the CGAL 2D Convex Hull package provided under the GPL-3.0+ license"""

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _CGAL_Convex_hull_2
else:
    import _CGAL_Convex_hull_2

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


import CGAL.CGAL_Kernel

def convex_hull_2(range, result):
    return _CGAL_Convex_hull_2.convex_hull_2(range, result)

def ch_bykat(range, result):
    return _CGAL_Convex_hull_2.ch_bykat(range, result)

def ch_eddy(range, result):
    return _CGAL_Convex_hull_2.ch_eddy(range, result)

def ch_graham_andrew(range, result):
    return _CGAL_Convex_hull_2.ch_graham_andrew(range, result)

def ch_melkman(range, result):
    return _CGAL_Convex_hull_2.ch_melkman(range, result)

def lower_hull_points_2(range, result):
    return _CGAL_Convex_hull_2.lower_hull_points_2(range, result)

def upper_hull_points_2(range, result):
    return _CGAL_Convex_hull_2.upper_hull_points_2(range, result)

def ch_jarvis(range, result):
    return _CGAL_Convex_hull_2.ch_jarvis(range, result)

def ch_akl_toussaint(range, result):
    return _CGAL_Convex_hull_2.ch_akl_toussaint(range, result)

def ch_graham_andrew_scan(range, result):
    return _CGAL_Convex_hull_2.ch_graham_andrew_scan(range, result)

def ch_jarvis_march(range, start_p, stop_p, result):
    return _CGAL_Convex_hull_2.ch_jarvis_march(range, start_p, stop_p, result)

def is_ccw_strongly_convex_2(range):
    return _CGAL_Convex_hull_2.is_ccw_strongly_convex_2(range)

def is_cw_strongly_convex_2(range):
    return _CGAL_Convex_hull_2.is_cw_strongly_convex_2(range)

def ch_n_point(range, n):
    return _CGAL_Convex_hull_2.ch_n_point(range, n)

def ch_s_point(range, s):
    return _CGAL_Convex_hull_2.ch_s_point(range, s)

def ch_e_point(range, e):
    return _CGAL_Convex_hull_2.ch_e_point(range, e)

def ch_w_point(range, w):
    return _CGAL_Convex_hull_2.ch_w_point(range, w)

def ch_we_point(range, w, e):
    return _CGAL_Convex_hull_2.ch_we_point(range, w, e)

def ch_ns_point(range, n, s):
    return _CGAL_Convex_hull_2.ch_ns_point(range, n, s)

def ch_nswe_point(range, n, s, w, e):
    return _CGAL_Convex_hull_2.ch_nswe_point(range, n, s, w, e)

