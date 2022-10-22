import os

if os.getenv("DISABLE_FUSE", False):
    memory_efficient_fusion = lambda x: x
else:
    from functorch.compile import memory_efficient_fusion


def aot_fuse(*args, **kwargs):
    return memory_efficient_fusion(*args, **kwargs)
