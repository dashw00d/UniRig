from .spec import ModelSpec

def get_model(**kwargs) -> ModelSpec:
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'unirig_ar':
        from .unirig_ar import UniRigAR
        return UniRigAR(**kwargs)
    if __target__ == 'unirig_skin':
        from .unirig_skin import UniRigSkin
        return UniRigSkin(**kwargs)
    raise AssertionError(f"expect: [unirig_ar,unirig_skin], found: {__target__}")
