def call_mee(backend=0, *args, **kwargs):
    """
    decide backend and return RCWA mee instance

    Args:
        backend: decide backend
        *args: passed to RCWA instance
        **kwargs: passed to RCWA instance

    Returns:
        RCWA: RCWA mee instance

    """
    if backend == 0:
        from .on_numpy.mee import MeeNumpy
        mee = MeeNumpy(backend=backend, *args, **kwargs)
    elif backend == 2:
        from .on_torch.mee import MeeTorch
        mee = MeeTorch(backend=backend, *args, **kwargs)
    else:
        raise ValueError
    return mee