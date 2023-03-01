from abc import ABCMeta


# This metaclass allows abstract classes with hooks
class ABCHookAfterInit(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super(ABCHookAfterInit, cls).__call__(*args, **kwargs)
        try:
            instance.hook_after_init()
        except AttributeError:
            print('No hook after init defined')
        return instance


def reversed_zip(*args):
    return zip(*map(reversed, args))
