from collections.abc import MutableMapping

class House(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._values = dict()
        self.update(dict(*args, **kwargs))
    
    def __getitem__(self, key):
      return self._values[key]
  
    def __getattr__(self, attr):
        return self.get(attr)

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]
    
    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __contains__(self, item):
        if item in self._values.keys():
            return True
        return False
    
    def pop(self, k):
        return self._values.pop(k)
    
    def keys(self):
        return self._values.keys()
    
    def items(self):
        return self._values.items()
    
    def values(self):
        return self._values.values()
    


        
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
    