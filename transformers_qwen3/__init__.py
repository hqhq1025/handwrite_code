# 这是 Qwen3 模块的惰性导入入口, 延迟加载配置与建模代码, 减少初始导入开销

from typing import TYPE_CHECKING

# 这里表示从"向上三级目录"导入对应的模块
from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

# 类型检查阶段
if TYPE_CHECKING:
    #导入当前文件夹下的两个文件, 让检查器指导模块导出的名称, 用于补全和类型判断
    from .configuration_qwen3 import *
    from .modeling_qwen3 import *

# 类型检查之外(实际运行阶段)
else:
    # 拿到 python 的模块表, 这是 python 维护的"已加载模块字典"
    # 后续通过替换为 lazy_module 实例, 实现访问前的"懒加载"
    import sys
    
    _file = globals()["__file__"]  # 记录当前文件路径
    
    # 把当前模块在 module 中的条目替换成 lazy_module实例
    # 在加载器真正用到某属性时, 才执行实际导入
    # 可以缩短初次导入时间
    sys.module[__name__] = _LazyModule(__name__, globals()["__file__"], define_import_structure, module_spec = __spec__)