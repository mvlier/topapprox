'''
Basin Hierarchy Tree class
'''

# TODO
class BasinHierarchyTree():
    def __init__(self) -> None:
        self.parent = None
        self.children = None
        self.persistent_children = None
        self.positive_pers = None # 1D numpy array to store the list of vertices with positive persistence, apart from the root
        self.birth = None
    
