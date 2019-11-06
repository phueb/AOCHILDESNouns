import attr


@attr.s
class PrepParams(object):
    reverse = attr.ib(default=False)
    num_types = attr.ib(default=None)  # None results in all types included in vocabulary
    num_parts = attr.ib(default=2)
    num_iterations = attr.ib(default=([1, 1]))
    batch_size = attr.ib(default=1)
    context_size = attr.ib(default=7)
    num_evaluations = attr.ib(default=10)