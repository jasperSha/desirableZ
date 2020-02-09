#init Property with specified dict
class PropertyZest:
    def __init__(self, categories):
        for (key, value) in categories.items():
            setattr(self, key, value)
            
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def items(self):
        return self.__dict__.items()
    
    def keys(self):
        return self.__dict__.keys()
    
    def values(self):
        return self.__dict__.values()
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):
        return repr(self.__dict__)
    
    def __len__(self):
        return len(self.__dict__)
    
    def __delitem__(self, key):
        del self.__dict__[key]
        
    def __iter__(self):
        return iter(self.__dict__)
    
    def clear(self):
        return self.__dict__.clear()
    
    def copy(self):
        return self.__dict__.copy()
        
class School:
    def __init__(self, categories):
        for (key, value) in categories.items():
            setattr(self, key, value)
    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getitem__(self, key):
        return self.__dict__[key]
    def __repr__(self):
        return repr(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    def __delitem__(self, key):
        del self.__dict__[key]
    def __iter__(self):
        return iter(self.__dict__)
    def clear(self):
        return self.__dict__.clear()
    def copy(self):
        return self.__dict__.copy()
    