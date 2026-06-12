# Register this vendored package under the short alias ``_c3`` so that
# lazy imports like ``from _c3.model.vfm.mot... import ...`` inside
# cosmos3_vla.py and transforms resolve correctly via sys.modules.
import sys as _sys
import fluxvla.models.third_party_models.cosmos3 as _self  # noqa: F401

_sys.modules.setdefault('_c3', _self)
