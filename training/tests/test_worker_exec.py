import pytest
from rlm_train.repl.subprocess import SubprocessReplBackend


@pytest.fixture
async def backend():
    b = SubprocessReplBackend()
    await b.start("http://127.0.0.1:9", "test", 1)
    yield b
    await b.stop()


async def test_blocks_are_compiled_with_source(backend: SubprocessReplBackend) -> None:
    # `@triton.jit` reads the source of the decorated function via `inspect`,
    # which only works if the block was compiled under a registered filename.
    result = await backend.execute(
        "import inspect\ndef f(x):\n    return x + 1\nprint(inspect.getsource(f))"
    )
    assert result.exception is None
    assert result.stdout == "def f(x):\n    return x + 1\n\n"


async def test_exception_is_reported(backend: SubprocessReplBackend) -> None:
    assert (await backend.execute("1/0")).exception == "ZeroDivisionError: division by zero"
    assert (await backend.execute("def g(:\n  pass")).exception.startswith("SyntaxError:")


async def test_set_local_and_submit_idiom(backend: SubprocessReplBackend) -> None:
    await backend.set_local("kernel_src", "src")
    result = await backend.execute('answer["content"] = kernel_src\nanswer["ready"] = True')
    assert result.final_answer == "src"
