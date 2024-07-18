def test_version():
    # Check that we can import the version number
    from steering_bench import __version__

    assert isinstance(__version__, str)
    assert __version__ != ""
