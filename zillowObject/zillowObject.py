#init Property with specified dict
class PropertyZest:
    def __init__(self, categories):
        for (key, value) in categories.items():
            setattr(self, key, value)
        
