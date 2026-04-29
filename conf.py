# -- Project information -----
import os
from pathlib import Path

project = "AI系统工程师培训教程"
subtitle = "面向零基础入门的AI系统全面培训教材"
copyright = "2024"
language = "en"  # 临时改为英文以避免 polyglossia 问题
master_doc = "index"

# -- LaTeX/PDF configuration ------
latex_documents = [
    (master_doc, "AI-System-Training.tex", project, subtitle, "manual"),
]

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "utf8": True,
    "toc_depth": 3,
    "preamble": r"""
\usepackage{fontspec}
\setmainfont{STSong}
\setsansfont{STHeiti}
\setmonofont{STSong}
""",
}

latex_theme = "sphinx"

# 使用 xelatex 引擎支持系统字体和中文
latex_engine = "xelatex"

latex_keep_old_macro_names = False
latex_use_xindy = False

# -- General configuration ------

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_togglebutton",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}

nitpicky = True
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

# -- Options for HTML output ----

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_title = "AI系统工程师培训教程"
html_copy_source = True
html_last_updated_fmt = ""

html_static_path = ["_static"]
nb_execution_mode = "off"

html_theme_options = {
    "path_to_docs": "",
    "repository_url": "https://github.com/your-org/AI-System-Training",
    "repository_branch": "main",
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "thebe": False,
    },
    "use_edit_page_button": False,
    "use_source_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 3,
    "home_page_in_toc": True,
    "extra_footer": "<p>本教材仅供内部培训使用</p>",
}

# -- Learning Path Navigation --
learning_goals = {
    "目标读者": "零基础新人（从半导体硬件转向AI软件）",
    "学习方式": "按学习路径系统性学习",
    "内容深度": "面向理解，侧重概念原理",
}
