import os

if os.getenv("DISABLE_FUNCTORCH", "false").lower() in ["1", "true", "yes"]:
    memory_efficient_fusion = lambda x: x
else:
    from functorch.compile import memory_efficient_fusion


def aot_fuse(*args, **kwargs):
    return memory_efficient_fusion(*args, **kwargs)
