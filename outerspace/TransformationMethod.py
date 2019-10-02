from abc import ABC, abstractmethod


class TransformationMethod(ABC):
    @abstractmethod
    def get_widgets(self):
        pass

    @abstractmethod
    def get_current_params(self, widgets):
        pass

    @abstractmethod
    def run_transformation(self, X, y, transformation_params, callback):
        pass
