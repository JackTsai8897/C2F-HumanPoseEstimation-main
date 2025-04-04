# setup_windows.py
import os
from os.path import join as pjoin
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from distutils.command.build_ext import build_ext
from distutils.spawn import spawn, find_executable
import sys

PATH = os.environ.get('PATH')

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    if 'CUDA_PATH' in os.environ:
        home = os.environ['CUDA_PATH']
    else:
        for cuda_version in ['12.1', '12.0', '11.8']:
            path = rf'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{cuda_version}'
            if os.path.exists(path):
                home = path
                break
        else:
            raise EnvironmentError('CUDA not found. Please install CUDA and set CUDA_PATH environment variable')

    cudaconfig = {
        'home': home,
        'nvcc': pjoin(home, 'bin', 'nvcc.exe'),
        'include': pjoin(home, 'include'),
        'lib64': pjoin(home, 'lib', 'x64')
    }

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f'The CUDA {k} path could not be located in {v}')

    return cudaconfig

CUDA = locate_cuda()

class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        if hasattr(self.compiler, '_c_extensions'):
            self.compiler._c_extensions.append('.cu')  # needed for Windows
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)
    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        '''
            Perform any CUDA specific customizations before actually launching
            compile/link etc. commands.
        '''
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        elif self.compiler.compiler_type == 'msvc':
            # There are several things we need to do to change the commands
            # issued by MSVCCompiler into one that works with nvcc. In the end,
            # it might have been easier to write our own CCompiler class for
            # nvcc, as we're only interested in creating a shared library to
            # load with ctypes, not in creating an importable Python extension.
            # - First, we replace the cl.exe or link.exe call with an nvcc
            #   call. In case we're running Anaconda, we search cl.exe in the
            #   original search path we captured further above -- Anaconda
            #   inserts a MSVC version into PATH that is too old for nvcc.
            cmd[:1] = ['nvcc', '--compiler-bindir',
                       os.path.dirname(find_executable("cl.exe", PATH))
                       or cmd[0]]
            # - Secondly, we fix a bunch of command line arguments.
            for idx, c in enumerate(cmd):
                # create .dll instead of .pyd files
                #if '.pyd' in c: cmd[idx] = c = c.replace('.pyd', '.dll')  #20160601, by MrX
                # replace /c by -c
                if c == '/c': cmd[idx] = '-c'
                # replace /DLL by --shared
                elif c == '/DLL': cmd[idx] = '--shared'
                # remove --compiler-options=-fPIC
                elif '-fPIC' in c: del cmd[idx]
                # replace /Tc... by ...
                elif c.startswith('/Tc'): cmd[idx] = c[3:]
                # replace /Fo... by -o ...
                elif c.startswith('/Fo'): cmd[idx:idx+1] = ['-o', c[3:]]
                # replace /LIBPATH:... by -L...
                elif c.startswith('/LIBPATH:'): cmd[idx] = '-L' + c[9:]
                # replace /OUT:... by -o ...
                elif c.startswith('/OUT:'): cmd[idx:idx+1] = ['-o', c[5:]]
                # remove /EXPORT:initlibcudamat or /EXPORT:initlibcudalearn
                elif c.startswith('/EXPORT:'): del cmd[idx]
                # replace cublas.lib by -lcublas
                elif c == 'cublas.lib': cmd[idx] = '-lcublas'
            # - Finally, we pass on all arguments starting with a '/' to the
            #   compiler or linker, and have nvcc handle all other arguments
            if '--shared' in cmd:
                pass_on = '--linker-options='
                # we only need MSVCRT for a .dll, remove CMT if it sneaks in:
                cmd.append('/NODEFAULTLIB:libcmt.lib')
            else:
                pass_on = '--compiler-options='
            cmd = ([c for c in cmd if c[0] != '/'] +
                   [pass_on + ','.join(c for c in cmd if c[0] == '/')])
            # For the future: Apart from the wrongly set PATH by Anaconda, it
            # would suffice to run the following for compilation on Windows:
            # nvcc -c -O -o <file>.obj <file>.cu
            # And the following for linking:
            # nvcc --shared -o <file>.dll <file1>.obj <file2>.obj -lcublas
            # This could be done by a NVCCCompiler class for all platforms.
        spawn(cmd, search_path, verbose, dry_run)

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

ext_modules = cythonize([
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        include_dirs=[numpy_include],
        define_macros=define_macros,
        language="c++"
    ),
    Extension(
        "gpu_nms",
        ["nms_kernel.cu", "gpu_nms.pyx"],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[],
        include_dirs=[numpy_include, CUDA['include']],
        define_macros=define_macros,
        extra_compile_args={
            'gcc': [],
            'nvcc': [
                '-arch=sm_89',
                '-gencode=arch=compute_89,code=sm_89',
                '--ptxas-options=-v',
                '-c',
                '--compiler-options',
                '"/MD"'
            ]
        }
    )
])

setup(
    name='nms',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CUDA_build_ext},
)