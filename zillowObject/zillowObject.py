#init Property with specified dict
class PropertyZest:
    def __init__(self, categories):
        for (key, value) in categories.items():
            setattr(self, key, value)
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
class School:
    def __init__(self, categories):
        for (key, value) in categories.items():
            setattr(self, key, value)
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)