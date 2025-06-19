
from setuptools import setup, find_packages

setup(
    name="data-generate-library",  # Название пакета
    version="0.1",      # Версия библиотеки
    author="Belyashov Artyom",  # Ваше имя
    author_email="me@belyashow.ru",  # Ваш email
    description="A library for simulating spinal cord models",  # Краткое описание
    long_description=open("README.md").read(),  # Полное описание из README.md
    long_description_content_type="text/markdown",  # Формат описания
    url="https://github.com/Belyashov-Artyom-MIET/data-generate-library",   # Ссылка на репозиторий
    license="MIT",  # Лицензия
    packages=find_packages(),  # Автоматическое нахождение всех пакетов
    include_package_data=True,  # Включение не-Python файлов (например, JSON)
    package_data={
        "data-generate-library": ["*.json"],  # Указываем, что нужно включить все JSON-файлы
    },
    install_requires=[
        "numpy==1.26.4",
        "pandas",
        "matplotlib",
        "netpyne",
        "neuron",
        "random",
        "pickle",
    ],  # Зависимости библиотеки
)
