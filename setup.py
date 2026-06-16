import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

# A CMakeExtension that simply records the name and sourcedir
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            import subprocess
            subprocess.check_output([sys.executable, "-m", "cmake", "--version"])
        except subprocess.CalledProcessError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        import pybind11
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DCMAKE_BUILD_TYPE=Release"
        ]

        build_args = []

        env = os.environ.copy()
        
        build_temp_ext = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp_ext):
            os.makedirs(build_temp_ext)

        import subprocess
        
        # Use python -m cmake to avoid wrapper script issues
        cmake_cmd = [sys.executable, "-m", "cmake"]
        
        subprocess.check_call(cmake_cmd + [ext.sourcedir] + cmake_args, cwd=build_temp_ext, env=env)
        subprocess.check_call(cmake_cmd + ["--build", "."] + build_args, cwd=build_temp_ext)

setup(
    name="quant_os_core",
    version="0.1",
    author="Quant OS",
    description="High-performance C++ core via pybind11",
    ext_modules=[
        CMakeExtension("cpp_features", "src/cpp_core"),
        CMakeExtension("cpp_zmq_consumer", "src/data_gateway"),
        CMakeExtension("cpp_timeseries_db", "src/data_gateway"),
        CMakeExtension("cpp_order_book", "src/execution/order_manager"),
        CMakeExtension("cpp_sor", "src/execution/smart_order_router")
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
