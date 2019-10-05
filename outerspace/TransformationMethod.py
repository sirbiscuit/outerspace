from abc import ABC, abstractmethod


class TransformationMethod(ABC):
    @abstractmethod
    def get_widgets(self):
        ''' Create and return widgets for manipulating the parameters of the
        transformation method.

        Returns
        -------
        OrderedDict[str -> ipywidgets.Widget]
            A dict that represents a mapping of labels to widgets (of the
            ipywidgets library). The values will be rendered in the order that
            is given by the items() iterator of the dictionary. For that reason
            it is sensible to return an instance of Pythons OrderedDict. The
            keys are used to identify and find widgets e.g. in
            get_current_params.
        '''
        pass

    @abstractmethod
    def get_current_params(self, widgets):
        ''' Returns the current parameters of the transformation method from
        the current state of the widgets.

        Returns
        -------
        dict
            The parameters of the transformation method that will be used in
            run_transformation.
        '''
        pass

    @abstractmethod
    def run_transformation(self, X, y, transformation_params, callback):
        ''' Executes the actual transformation method.

        Parameters
        ----------
        X : np.ndarray of shape (n_examples, n_features)
            The feature variables.
        y : array_like of shape (n_examples,)
            The target variable used for coloring the data points.
        transformation_params : dict
            Parameters for the transformation method.
        callback : Callable[[command, iteration, payload]]
            A callback that is intended for providing feedback to the user.
            Multiple different commands are available:
                * start: the transformation method was initialized and is
                    running. The payload is expected to be a dict
                    { error_metrics } where error_metrics is an array of
                    dicts { name, label }. Each entry creates a widget
                    with description text "label". The field "name" is an
                    identifier that will be useful in the "embedding" command.
                * embedding: a new embedding is available. The payload is a
                    dict { embedding, error_metrics } where embedding is a
                    numpy.ndarray and error_metrics is a dict.
                * error: an error occured and payload contains the error
                    message.
                * status: for providing other feedback to the user. The
                    message in the payload is displayed.
        '''
        pass
