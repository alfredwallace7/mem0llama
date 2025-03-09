from setuptools import setup, find_packages

setup(
    name="mem0llama",
    version="0.1.3",
    description="A specialized fork of Mem0 optimized for small, local Large Language Models (LLMs) running through Ollama.",
    author="Alfred WALLACE",
    author_email="alfred.wallace@netcraft.fr",
    license="Apache-2.0",
    url="https://github.com/alfredwallace7/mem0llama",
    packages=find_packages(include=["mem0llama", "mem0llama.*"]),
    install_requires=[
        "qdrant-client>=1.9.1",
        "pydantic>=2.7.3",
        "openai>=1.33.0",
        "pytz>=2024.1",
        "sqlalchemy>=2.0.31",
        "langchain-community>=0.3.1",
        "langchain-neo4j>=0.3.1",
        "rank-bm25>=0.2.2",
        "psycopg2-binary>=2.9.10",
        "python-dotenv>=1.0.0",
        "litellm>=1.0.0",
        "rich>=13.0.0",
        "ollama>=0.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <4.0",
    keywords=["memory", "llm", "ai", "agents", "qdrant", "neo4j", "ollama"]
)
