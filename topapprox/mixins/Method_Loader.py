import importlib
import warnings

class MethodLoaderMixin:
    def load_method(self, method, package, *, iter_vertex=False):
        if method == "cpp":
            try:
                module = importlib.import_module('.link_reduce_cpp', package=package)
                if iter_vertex:
                    self._link_reduce = getattr(module, '_link_reduce_vertices_cpp')
                else:
                    self._link_reduce = getattr(module, '_link_reduce_cpp')
                # self.compute_descendants = getattr(module, 'compute_descendants_cpp')
            except Exception as e:
                warnings.warn(e + """\n Falling back to python (slower) since C++ version could not be loaded\n
                              For using numba version one can set method='numba'""", UserWarning)
                method = "python"
        elif method == "numba":
            try:
                module = importlib.import_module('.link_reduce_numba', package=package)
                if iter_vertex:
                    self._link_reduce = getattr(module, 'link_reduce_vertices_wrapper')
                else:
                    self._link_reduce = getattr(module, 'link_reduce_wrapper')
                # self.compute_descendants = getattr(module, 'compute_descendants_numba')
            except Exception as e:
                warnings.warn(e + """\n Falling back to python since numba version could not be loaded\n
                              For using C++ version (recommended for performance) one can set method='cpp'""", UserWarning)
                method = "python"
        elif method != "python":
            raise ValueError(f"Unknown method: {method}")
        
        if method == "python":
            module = importlib.import_module('.link_reduce', package=package)
            if iter_vertex:
                self._link_reduce = getattr(module, '_link_reduce_vertices')
                # self.compute_descendants = getattr(module, 'compute_descendants')
            else:
                self._link_reduce = getattr(module, '_link_reduce')

        return method
