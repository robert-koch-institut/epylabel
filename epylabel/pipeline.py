"""
A class representing a data transformation pipeline.

This class allows chaining multiple transformation steps to be applied
sequentially on the input data.

:param steps:
    A list of transformation steps in the order they should be applied.
"""
from epylabel.labeler import Transformation


class Pipeline(Transformation):
    """
    A class representing a data transformation pipeline.

    This class allows chaining multiple transformation steps to be applied
    sequentially on the input data.

    :param steps:
        A list of transformation steps in the order they should be applied.
    """

    def __init__(self, steps):
        """
        Initialize a Pipeline instance.

        :param steps:
            A list of transformation steps in the order they should be applied.
        """
        self.steps = steps

    def transform(self, *args):
        """
        Apply the transformation pipeline to the input data.

        :param args:
            Input data that will be sequentially transformed by the pipeline steps.

        :return:
            Transformed output data after applying all pipeline steps.
        """
        out = self.steps[0].transform(*args)
        for i in range(1, len(self.steps)):
            step = self.steps[i]
            out = step.transform(out)
        return out

    def __call__(self, *args):
        """
        Alias for the transform method, allowing the pipeline to be
        called as a function.

        :param args:
            Input data that will be sequentially transformed by the
            pipeline steps.

        :return:
            Transformed output data after applying all pipeline steps.
        """
        return self.transform(*args)
